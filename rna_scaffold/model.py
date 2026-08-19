from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class ScaffoldModelOutput:
    token_logits: torch.Tensor
    length_logits: torch.Tensor
    position_logits: torch.Tensor


@dataclass(frozen=True)
class DenoisingLosses:
    total_loss: torch.Tensor
    base_loss: torch.Tensor
    length_loss: torch.Tensor
    position_loss: torch.Tensor


def compute_denoising_losses(
    output: ScaffoldModelOutput,
    target_base_ids: torch.Tensor,
    fixed_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    target_length: torch.Tensor,
    motif_start: torch.Tensor,
    prediction_mask: torch.Tensor | None = None,
    length_loss_weight: float = 0.25,
    position_loss_weight: float = 0.25,
    label_smoothing: float = 0.0,
) -> DenoisingLosses:
    scaffold_mask = (
        prediction_mask.bool()
        if prediction_mask is not None
        else attention_mask.bool() & ~fixed_mask.bool()
    )
    if not scaffold_mask.any():
        raise ValueError("every batch must contain at least one scaffold position")
    base_loss = F.cross_entropy(
        output.token_logits[scaffold_mask],
        target_base_ids[scaffold_mask],
        label_smoothing=label_smoothing,
    )
    length_loss = F.cross_entropy(output.length_logits, target_length.long())
    position_loss = F.cross_entropy(output.position_logits, motif_start.long())
    total_loss = base_loss + length_loss_weight * length_loss + position_loss_weight * position_loss
    return DenoisingLosses(total_loss, base_loss, length_loss, position_loss)


def restore_fixed_tokens(
    proposed_ids: torch.Tensor,
    original_ids: torch.Tensor,
    fixed_mask: torch.Tensor,
) -> torch.Tensor:
    if proposed_ids.shape != original_ids.shape or proposed_ids.shape != fixed_mask.shape:
        raise ValueError("proposed_ids, original_ids, and fixed_mask must have identical shapes")
    return torch.where(fixed_mask.bool(), original_ids, proposed_ids)


class MotifDenoisingTransformer(nn.Module):
    """Bidirectional RNA scaffold model with token, length, and position heads."""

    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_length: int = 512,
        activation_checkpointing: bool = False,
        pretrained_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.activation_checkpointing = activation_checkpointing
        self.pretrained_encoder = pretrained_encoder
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.pretrained_projection = (
            nn.Linear(int(pretrained_encoder.output_dim), d_model, bias=False)
            if pretrained_encoder is not None
            else None
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(d_model)
        self.token_head = nn.Linear(d_model, 4)
        self.length_head = nn.Linear(d_model, max_length + 1)
        self.position_head = nn.Linear(d_model, max_length)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.pretrained_encoder is not None and getattr(
            self.pretrained_encoder, "_rna_scaffold_frozen", False
        ):
            self.pretrained_encoder.eval()
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> ScaffoldModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, length]")
        batch_size, length = input_ids.shape
        if length > self.max_length:
            raise ValueError(f"input length {length} exceeds max_length {self.max_length}")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)
        attention_mask = attention_mask.bool()
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        if self.pretrained_encoder is not None:
            pretrained_features = self.pretrained_encoder(input_ids, attention_mask)
            hidden = hidden + self.pretrained_projection(pretrained_features)
        padding_mask = ~attention_mask
        if self.activation_checkpointing and self.training:
            for layer in self.encoder.layers:
                hidden = checkpoint(
                    layer,
                    hidden,
                    src_key_padding_mask=padding_mask,
                    use_reentrant=False,
                )
            if self.encoder.norm is not None:
                hidden = self.encoder.norm(hidden)
        else:
            hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        hidden = self.final_norm(hidden)
        weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return ScaffoldModelOutput(
            token_logits=self.token_head(hidden),
            length_logits=self.length_head(pooled),
            position_logits=self.position_head(pooled),
        )
