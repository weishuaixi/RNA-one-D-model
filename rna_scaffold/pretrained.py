from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class PretrainedEncoderConfig:
    kind: str = "none"
    checkpoint: str | None = None
    freeze: bool = True


def _import_fm():
    import fm

    return fm


class RnaFmEncoder(nn.Module):
    """Align official RNA-FM residue representations with project token positions."""

    output_dim = 640
    representation_layer = 12

    def __init__(self, model: nn.Module, alphabet: Any) -> None:
        super().__init__()
        self.model = model
        self.alphabet = alphabet
        self._project_to_fm = {
            0: alphabet.padding_idx,
            3: alphabet.mask_idx,
            8: alphabet.get_idx("A"),
            9: alphabet.get_idx("U"),
            10: alphabet.get_idx("C"),
            11: alphabet.get_idx("G"),
        }

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, length = input_ids.shape
        tokens = torch.full(
            (batch_size, length + 2),
            self.alphabet.padding_idx,
            dtype=torch.long,
            device=input_ids.device,
        )
        tokens[:, 0] = self.alphabet.cls_idx
        residue_tokens = tokens[:, 1 : length + 1]
        for source_id, fm_id in self._project_to_fm.items():
            residue_tokens.masked_fill_(input_ids.eq(source_id), fm_id)
        sequence_lengths = attention_mask.long().sum(dim=1)
        tokens[torch.arange(batch_size, device=input_ids.device), sequence_lengths + 1] = self.alphabet.eos_idx
        results = self.model(tokens, repr_layers=[self.representation_layer])
        return results["representations"][self.representation_layer][:, 1 : length + 1]


def load_rna_fm(checkpoint: str | None) -> nn.Module:
    try:
        fm = _import_fm()
    except ImportError as error:
        raise RuntimeError(
            "RNA-FM is optional. Install it in the server environment with "
            "`pip install rna-fm`, or set pretrained.kind to 'none'."
        ) from error
    if checkpoint:
        model, alphabet = fm.pretrained.load_model_and_alphabet_local(Path(checkpoint))
    else:
        model, alphabet = fm.pretrained.rna_fm_t12()
    return RnaFmEncoder(model, alphabet)


def build_pretrained_encoder(config: PretrainedEncoderConfig | dict[str, Any] | None) -> nn.Module | None:
    if config is None:
        return None
    if isinstance(config, dict):
        config = PretrainedEncoderConfig(**config)
    if config.kind == "none":
        return None
    if config.kind != "rna_fm":
        raise ValueError(f"unsupported pretrained encoder kind: {config.kind!r}")
    encoder = load_rna_fm(config.checkpoint)
    if config.freeze:
        encoder.requires_grad_(False)
        encoder.eval()
        encoder._rna_scaffold_frozen = True
    return encoder


def pretrained_metadata(config: PretrainedEncoderConfig | dict[str, Any] | None) -> dict[str, Any]:
    if config is None:
        config = PretrainedEncoderConfig()
    elif isinstance(config, dict):
        config = PretrainedEncoderConfig(**config)
    metadata = asdict(config)
    checkpoint = Path(config.checkpoint) if config.checkpoint else None
    metadata["checkpoint_sha256"] = _sha256(checkpoint) if checkpoint and checkpoint.is_file() else None
    return metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
