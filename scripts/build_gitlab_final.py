"""Package verified final V5 weights, runtime dependency weights and source."""

import hashlib
import json
import shutil
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from build_gitlab_source import ROOT, build

BACKUP = "V5-BEST-DOWNLOAD-20260910-042140-147668.tar.gz"
BACKUP_SHA = "255ec9de3319a3d92ea090f98d5ef351859c2355cf9c810e7dd8352dc2131dab"
CHECKPOINT = "checkpoints_scaffold_linker_v5/rna-linker-v5-34-1.3714.ckpt"
CHECKPOINT_SHA = "2a6bab7bfb4afc145c2309c38798e2db43f085d628c291ab4004b33932546ced"
RNAFM = ".cache/torch/hub/checkpoints/RNA-FM_pretrained.pth"
RNAFM_SHA = "5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99"


def sha(path):
    with path.open("rb") as stream:
        return stream_sha(stream)


def stream_sha(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def main():
    for name, expected in ((BACKUP, BACKUP_SHA), (RNAFM, RNAFM_SHA)):
        if sha(ROOT / name) != expected:
            raise ValueError(f"Input checksum mismatch: {name}")
    destination = ROOT / "exports" / ("RNA-V5-IGEM-FINAL-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    destination.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "source.zip"
        build(source)
        with zipfile.ZipFile(source) as archive:
            archive.extractall(destination)  # Only the builder's verified allowlist.
    manifest_name = "checkpoints_scaffold_linker_v5/training_manifest.json"
    with tarfile.open(ROOT / BACKUP) as archive:
        for name in (CHECKPOINT, manifest_name):
            member = archive.getmember(name)
            if not member.isfile():
                raise ValueError(f"Not a regular archive member: {name}")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
        original_config = archive.extractfile("configs/train_scaffold_5090_linker_v5.yaml").read()
    manifest = json.loads((destination / manifest_name).read_text(encoding="utf-8"))
    if sha(destination / CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("Final checkpoint mismatch")
    if hashlib.sha256(original_config).hexdigest() != manifest["config_sha256"]:
        raise ValueError("Training configuration provenance mismatch")
    if (destination / "configs/train_scaffold_5090_linker_v5.yaml").read_bytes() != original_config:
        raise ValueError("Packaged configuration differs from the trained configuration")
    (destination / RNAFM).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / RNAFM, destination / RNAFM)
    if sha(destination / RNAFM) != RNAFM_SHA:
        raise ValueError("Copied RNA-FM mismatch")
    files = sorted(p for p in destination.rglob("*") if p.is_file())
    digests = {p.relative_to(destination).as_posix(): sha(p) for p in files}
    report = {
        "kind": "final_model_and_source",
        "model_release": "V5",
        "includes_training_data": False,
        "includes_weights": True,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "rna_fm_sha256": RNAFM_SHA,
        "benchmark_evidence": "README historical results; replacement checkpoint not benchmarked here",
        "files": digests,
    }
    (destination / "FINAL_RELEASE_MANIFEST.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    output = destination.with_suffix(".zip")
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(destination).as_posix())
    with zipfile.ZipFile(output) as archive:
        for name, expected in digests.items():
            with archive.open(name) as stream:
                if stream_sha(stream) != expected:
                    raise ValueError(f"ZIP content mismatch: {name}")
    with output.with_suffix(".zip.sha256").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{sha(output)}  {output.name}\n")
    print(
        json.dumps(
            {"folder": str(destination), "archive": str(output), "bytes": output.stat().st_size, "sha256": sha(output)},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
