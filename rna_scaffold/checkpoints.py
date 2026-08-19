from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

from rna_scaffold.lightning_module import RnaScaffoldLitModule
from rna_scaffold.tokenizer import RnaTokenizer


class CheckpointCompatibilityError(RuntimeError):
    """Raised when a checkpoint cannot safely reconstruct this generator."""


@dataclass(frozen=True)
class LoadedScaffoldModel:
    model: RnaScaffoldLitModule
    tokenizer: RnaTokenizer
    checkpoint_sha256: str
    max_length: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scaffold_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> LoadedScaffoldModel:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"scaffold checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise CheckpointCompatibilityError("checkpoint root must be a mapping")
    hparams = payload.get("hyper_parameters")
    state_dict = payload.get("state_dict")
    if not isinstance(hparams, dict):
        raise CheckpointCompatibilityError("checkpoint is missing hyper_parameters")
    if not isinstance(state_dict, dict):
        raise CheckpointCompatibilityError("checkpoint is missing state_dict")

    tokenizer = RnaTokenizer()
    constructor = dict(hparams)
    constructor.setdefault("vocab_size", tokenizer.vocab_size)
    constructor.setdefault("pad_token_id", tokenizer.pad_token_id)
    try:
        model = RnaScaffoldLitModule(**constructor)
    except (TypeError, ValueError, RuntimeError) as error:
        raise CheckpointCompatibilityError(
            f"checkpoint hyper_parameters are incompatible: {error}"
        ) from error

    migrated_state = dict(state_dict)
    migrated_state.pop("model.confidence_head.weight", None)
    migrated_state.pop("model.confidence_head.bias", None)
    try:
        model.load_state_dict(migrated_state, strict=True)
    except RuntimeError as error:
        raise CheckpointCompatibilityError(f"checkpoint state_dict is incompatible: {error}") from error
    model.to(device)
    model.eval()
    return LoadedScaffoldModel(
        model=model,
        tokenizer=tokenizer,
        checkpoint_sha256=_sha256(checkpoint_path),
        max_length=model.model.max_length,
    )
