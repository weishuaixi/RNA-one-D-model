#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
sha256sum -c PACKAGE_SHA256SUMS
python -c 'import sys; assert sys.version_info[:2] == (3,10), "Please activate Python 3.10 first"'
python -m pip install 'setuptools>=69' wheel
python -m pip install 'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' 'torchaudio==2.7.1+cu128' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c constraints-v5-server.txt -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
export TORCH_HOME="$PWD/.cache/torch"
unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
python - <<'PY'
import torch
import yaml
from pathlib import Path
from rna_scaffold.datamodule import RnaScaffoldDataModule
from rna_scaffold.tokenizer import RnaTokenizer
from rna_scaffold.pretrained import build_pretrained_encoder
assert torch.cuda.is_available(), 'CUDA unavailable'
x = torch.ones(16,16,device='cuda',dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
cfg = yaml.safe_load(Path('configs/train_scaffold_5090_linker_v5.yaml').read_text())
dm = RnaScaffoldDataModule(tokenizer=RnaTokenizer(), **cfg['data'])
dm.setup('fit')
encoder = build_pretrained_encoder(cfg['model']['pretrained'])
print('READY:', torch.__version__, torch.cuda.get_device_name(0))
print('RNA-FM loaded; dataset split verified. No training started.')
PY
python -m pip freeze > environment-installed.txt
