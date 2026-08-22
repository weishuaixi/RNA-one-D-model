from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

_SUPPORTED_ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "out_proj")


class LoRALinear(nn.Module):
    """Frozen linear plus ``scale * B(A(dropout(inputs)))``."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        self.base = base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        factory_kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.Linear(base.in_features, self.rank, bias=False, **factory_kwargs)
        self.lora_B = nn.Linear(self.rank, base.out_features, bias=False, **factory_kwargs)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.lora_B(self.lora_A(self.dropout(inputs)))
        return self.base(inputs) + self.scale * residual


def inject_lora_adapters(
    model: nn.Module,
    last_n_layers: int,
    rank: int,
    alpha: float,
    dropout: float,
) -> tuple[str, ...]:
    """Freeze ``model`` and adapt supported projections in its final layers."""
    if last_n_layers <= 0:
        raise ValueError("LoRA last_n_layers must be positive")
    if rank <= 0:
        raise ValueError("LoRA rank must be positive")
    if alpha <= 0:
        raise ValueError("LoRA alpha must be positive")
    if not 0 <= dropout < 1:
        raise ValueError("LoRA dropout must be in [0, 1)")

    model.requires_grad_(False)
    layer_stacks = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.ModuleList) and name.split(".")[-1] == "layers"
    ]
    targets: list[tuple[str, nn.Module, str, nn.Linear]] = []
    for stack_name, layers in layer_stacks:
        first_layer = max(0, len(layers) - int(last_n_layers))
        for layer_index in range(first_layer, len(layers)):
            layer = layers[layer_index]
            for relative_name, module in layer.named_modules():
                projection_name = relative_name.split(".")[-1]
                if (
                    isinstance(module, nn.Linear)
                    and projection_name in _SUPPORTED_ATTENTION_PROJECTIONS
                    and "self_attn" in relative_name.split(".")
                ):
                    parent_name, _, child_name = relative_name.rpartition(".")
                    parent = layer.get_submodule(parent_name) if parent_name else layer
                    prefix = f"{stack_name}." if stack_name else ""
                    full_name = f"{prefix}{layer_index}.{relative_name}"
                    targets.append((full_name, parent, child_name, module))
        if targets:
            break

    if not targets:
        candidates = [name for name, module in model.named_modules() if name and not list(module.children())]
        candidate_text = ", ".join(candidates) if candidates else "<none>"
        raise ValueError(
            "no supported RNA-FM attention projections were found; "
            f"candidate module names: {candidate_text}"
        )

    injected_names: list[str] = []
    for full_name, parent, child_name, base in targets:
        # RNA-FM's functional fast path reads projection weights directly and
        # cannot express per-example LoRA input dropout. Its built-in fallback
        # calls q/k/v/out modules, so switch only each adapted attention instance.
        if hasattr(parent, "enable_torch_version"):
            parent.enable_torch_version = False
        setattr(parent, child_name, LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout))
        injected_names.append(full_name)
    model._rna_scaffold_lora_targets = tuple(injected_names)
    return tuple(injected_names)


@dataclass(frozen=True)
class PretrainedEncoderConfig:
    kind: str = "none"
    checkpoint: str | None = None
    expected_checkpoint_sha256: str | None = None
    expected_checkpoint_sha256_env: str | None = "RNA_FM_EXPECTED_SHA256"
    mode: str = "frozen"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_last_n_layers: int = 4

    def __post_init__(self) -> None:
        if self.kind not in {"none", "rna_fm"}:
            raise ValueError(f"unsupported pretrained encoder kind: {self.kind!r}")
        if self.mode not in {"frozen", "lora"}:
            raise ValueError(f"unsupported RNA-FM mode: {self.mode!r}")
        if self.lora_rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.lora_alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if self.lora_last_n_layers <= 0:
            raise ValueError("LoRA last_n_layers must be positive")
        if self.expected_checkpoint_sha256 is not None:
            _validate_sha256(self.expected_checkpoint_sha256)


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
    config = _coerce_config(config)
    if config.kind == "none":
        return None
    expected_digest = _resolve_expected_checkpoint_sha256(config)
    actual_digest = None
    if config.checkpoint:
        checkpoint_path = Path(config.checkpoint)
        if checkpoint_path.is_file():
            actual_digest = _sha256(checkpoint_path)
            if actual_digest != expected_digest:
                raise ValueError(
                    "RNA-FM checkpoint SHA-256 mismatch: "
                    f"expected={expected_digest}, actual={actual_digest}"
                )
    encoder = load_rna_fm(config.checkpoint)
    encoder._rna_scaffold_expected_checkpoint_sha256 = expected_digest
    encoder._rna_scaffold_checkpoint_sha256 = actual_digest
    if config.mode == "frozen":
        encoder.requires_grad_(False)
        encoder.eval()
        encoder._rna_scaffold_frozen = True
    else:
        inject_lora_adapters(
            encoder,
            last_n_layers=config.lora_last_n_layers,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
        encoder.eval()
        encoder._rna_scaffold_lora = True
    return encoder


def set_lora_adapter_train_mode(model: nn.Module, mode: bool) -> None:
    """Keep the pretrained base deterministic while toggling adapter dropout."""
    model.eval()
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.training = mode
            module.dropout.train(mode)
            module.lora_A.train(mode)
            module.lora_B.train(mode)


def pretrained_metadata(
    config: PretrainedEncoderConfig | dict[str, Any] | None,
    encoder: nn.Module | None = None,
) -> dict[str, Any]:
    config = _coerce_config(config)
    metadata = asdict(config)
    checkpoint = Path(config.checkpoint) if config.checkpoint else None
    metadata["checkpoint_sha256"] = _sha256(checkpoint) if checkpoint and checkpoint.is_file() else None
    metadata["expected_checkpoint_sha256"] = (
        getattr(encoder, "_rna_scaffold_expected_checkpoint_sha256", None)
        if encoder is not None
        else config.expected_checkpoint_sha256
    )
    is_lora = config.kind == "rna_fm" and config.mode == "lora"
    metadata["lora_dropout_semantics"] = "input" if is_lora else None
    metadata["lora_projection_execution"] = "module_forward" if is_lora else None
    metadata["adapter_targets"] = list(
        getattr(encoder, "_rna_scaffold_lora_targets", ()) if encoder is not None else ()
    )
    parameters = list(encoder.parameters()) if encoder is not None else []
    metadata["parameter_counts"] = {
        "total": sum(parameter.numel() for parameter in parameters),
        "trainable": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
    }
    return metadata


def _coerce_config(
    config: PretrainedEncoderConfig | dict[str, Any] | None,
) -> PretrainedEncoderConfig:
    if config is None:
        return PretrainedEncoderConfig()
    if isinstance(config, PretrainedEncoderConfig):
        return config
    if "freeze" in config:
        raise ValueError("pretrained.freeze was removed in V2; use pretrained.mode")
    return PretrainedEncoderConfig(**config)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str) -> str:
    digest = str(value).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("expected RNA-FM checkpoint SHA-256 must be exactly 64 hex characters")
    return digest


def _resolve_expected_checkpoint_sha256(config: PretrainedEncoderConfig) -> str | None:
    if not config.checkpoint:
        return None
    configured = config.expected_checkpoint_sha256
    if configured is None and config.expected_checkpoint_sha256_env:
        configured = os.environ.get(config.expected_checkpoint_sha256_env)
    if configured is None:
        raise ValueError(
            "expected RNA-FM checkpoint SHA-256 is required; configure "
            "expected_checkpoint_sha256 or RNA_FM_EXPECTED_SHA256"
        )
    return _validate_sha256(configured)
