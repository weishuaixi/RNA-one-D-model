#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 was not found in PATH." >&2
  exit 127
fi

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        print(index, torch.cuda.get_device_name(index))
PY
