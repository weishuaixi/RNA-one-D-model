#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 was not found in PATH." >&2
  exit 127
fi

SEQUENCE_TABLE="data/processed/rna_merged_mmseqs80.csv"
CLUSTER_MANIFEST="outputs/mmseqs80_search/rna80_connected_components.tsv"
for required_file in "${SEQUENCE_TABLE}" "${CLUSTER_MANIFEST}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "ERROR: required canonical input is missing: ${required_file}" >&2
    exit 2
  fi
done

if [[ ! "${RNA_FM_EXPECTED_SHA256:-}" =~ ^[0-9a-fA-F]{64}$ ]]; then
  echo "ERROR: set RNA_FM_EXPECTED_SHA256 to the reviewed 64-hex RNA-FM weight digest." >&2
  exit 2
fi

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

export TORCH_HOME="${PROJECT_ROOT}/.cache/torch"
mkdir -p "${TORCH_HOME}" outputs/server_setup

python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is required for the authoritative A800 workflow")
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY

python - <<'PY'
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import yaml

from rna_scaffold.datamodule import RnaScaffoldDataModule
from rna_scaffold.tokenizer import RnaTokenizer

config = yaml.safe_load(Path("configs/train_scaffold_a800.yaml").read_text(encoding="utf-8"))
data_config = config["data"]
module = RnaScaffoldDataModule(tokenizer=RnaTokenizer(), **data_config)
module.setup("fit")
sources = {
    "sequence_table": Path(data_config["train_data"]),
    "cluster_manifest": Path(data_config["cluster_manifest"]),
}
audit = {
    "source_sha256": {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in sources.items()
    },
    "split_manifest": asdict(module.split_manifest),
    "data_audit": module.denoising_data_audit,
}
output = Path("outputs/server_setup/canonical_data_audit.json")
temporary = output.with_suffix(".json.tmp")
temporary.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
temporary.replace(output)
print(json.dumps(audit["source_sha256"], indent=2, sort_keys=True))
print("canonical_data_audit", output)
PY

python - <<'PY'
import hashlib
import importlib.metadata
import os
from pathlib import Path

import fm
import torch

fm.pretrained.rna_fm_t12()
checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / "RNA-FM_pretrained.pth"
if not checkpoint.is_file():
    raise SystemExit(f"ERROR: RNA-FM cache download did not create {checkpoint}")
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
expected_digest = os.environ["RNA_FM_EXPECTED_SHA256"].strip().lower()
if digest != expected_digest:
    raise SystemExit(
        f"ERROR: RNA-FM checkpoint SHA-256 mismatch: expected={expected_digest}, actual={digest}"
    )
expected = Path(".cache/torch/hub/checkpoints/RNA-FM_pretrained.pth").resolve()
if checkpoint.resolve() != expected:
    raise SystemExit(f"ERROR: RNA-FM cache path mismatch: {checkpoint.resolve()} != {expected}")
Path("outputs/server_setup/rna_fm_sha256.txt").write_text(
    f"{digest}  {checkpoint}\n",
    encoding="utf-8",
)
print("rna_fm_checkpoint", checkpoint)
print("rna_fm_sha256", digest)
print("rna_fm_version", importlib.metadata.version("rna-fm"))
PY

python scripts/verify_scaffold_release.py \
  --output outputs/server_setup/scaffold_release_audit.json
