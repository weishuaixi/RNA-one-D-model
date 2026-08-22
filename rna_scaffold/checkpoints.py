from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from rna_scaffold.geometry import validate_scaffold_length_limit
from rna_scaffold.tokenizer import RnaTokenizer

if TYPE_CHECKING:
    from rna_scaffold.lightning_module import RnaScaffoldLitModule


class CheckpointCompatibilityError(RuntimeError):
    """Raised when a checkpoint cannot safely reconstruct this generator."""


@dataclass(frozen=True)
class LoadedScaffoldModel:
    model: RnaScaffoldLitModule
    tokenizer: RnaTokenizer
    checkpoint_sha256: str
    max_length: int
    architecture_version: int
    pretrained_metadata: dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_scaffold_checkpoint_payload(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise CheckpointCompatibilityError("checkpoint root must be a mapping")
    architecture_version = payload.get("architecture_version")
    if type(architecture_version) is not int or architecture_version != 2:
        raise CheckpointCompatibilityError(
            "checkpoint architecture_version 2 is required; the model must be retrained for V2"
        )
    hparams = payload.get("hyper_parameters")
    if isinstance(hparams, dict) and "max_length" in hparams:
        try:
            validate_scaffold_length_limit(
                hparams["max_length"],
                field="checkpoint hyper_parameters.max_length",
            )
        except (TypeError, ValueError) as error:
            raise CheckpointCompatibilityError(str(error)) from error
    return payload


def _read_checkpoint_payload(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[Path, dict]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"scaffold checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    return checkpoint_path, validate_scaffold_checkpoint_payload(payload)


def validate_scaffold_checkpoint_version(path: str | Path) -> None:
    _read_checkpoint_payload(path, device="cpu")


def load_scaffold_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> LoadedScaffoldModel:
    from rna_scaffold.lightning_module import RnaScaffoldLitModule

    checkpoint_path, payload = _read_checkpoint_payload(path, device=device)
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

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise CheckpointCompatibilityError(f"checkpoint state_dict is incompatible: {error}") from error
    model.to(device)
    model.eval()
    raw_metadata = payload.get("pretrained_encoder")
    if isinstance(raw_metadata, dict):
        pretrained = dict(raw_metadata)
    else:
        raw_pretrained = hparams.get("pretrained")
        pretrained = dict(raw_pretrained) if isinstance(raw_pretrained, dict) else {"kind": "none"}
    pretrained.setdefault("kind", "none")
    pretrained.setdefault("mode", "frozen")
    if pretrained["kind"] == "none":
        pretrained["mode"] = "none"
    return LoadedScaffoldModel(
        model=model,
        tokenizer=tokenizer,
        checkpoint_sha256=_sha256(checkpoint_path),
        max_length=model.model.max_length,
        architecture_version=2,
        pretrained_metadata=pretrained,
    )
