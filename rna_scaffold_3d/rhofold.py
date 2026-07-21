from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from rna_scaffold_3d.rna_atoms import RNA_NUM_ATOMS
from rna_scaffold_3d.sequence import RNA3D_MASK_ID, RNA3D_PAD_ID


@dataclass(frozen=True)
class RhoFoldConfig:
    vocab_size: int = 6
    d_model: int = 256
    pair_dim: int = 128
    msa_dim: int = 128
    nhead: int = 8
    num_e2e_layers: int = 4
    num_structure_layers: int = 2
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_len: int = 2048
    num_atoms: int = RNA_NUM_ATOMS
    num_distance_bins: int = 32
    recycle_iters: int = 1
    sequence_loss_initial_weight: float = 0.1


class RhoFoldModel(nn.Module):
    """Trainable in-repository RhoFold-style RNA structure model.

    This is an internal architecture, not a wrapper around the upstream
    predictor. It follows the main RhoFold design shape: sequence/MSA features,
    pair representation, recurrent end-to-end refinement, recycling, and
    structure/confidence heads.
    """

    def __init__(self, config: RhoFoldConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = RhoFoldConfig(**kwargs)
        self.config = config
        self.num_atoms = config.num_atoms
        self.num_distance_bins = config.num_distance_bins
        self.recycle_iters = max(1, int(config.recycle_iters))
        if config.sequence_loss_initial_weight <= 0:
            raise ValueError("sequence_loss_initial_weight must be positive.")
        self.task_log_variances = nn.Parameter(
            torch.tensor([0.0, -math.log(config.sequence_loss_initial_weight)], dtype=torch.float32)
        )

        self.seq_embedder = SequenceEmbedder(config)
        self.msa_embedder = MSAEmbedder(config)
        self.pair_embedder = PairEmbedder(config)
        self.recycling = RecyclingEmbedder(config)
        self.e2eformer = nn.ModuleList([E2EformerBlock(config) for _ in range(config.num_e2e_layers)])
        self.structure_module = StructureModule(config)
        self.distogram_head = nn.Sequential(
            nn.LayerNorm(config.pair_dim),
            nn.Linear(config.pair_dim, config.pair_dim),
            nn.GELU(),
            nn.Linear(config.pair_dim, config.num_distance_bins),
        )
        self.plddt_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
        )
        self.sequence_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.vocab_size),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        msa_ids: torch.Tensor | None = None,
        msa_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if padding_mask is None:
            padding_mask = input_ids.eq(RNA3D_PAD_ID)

        seq = self.seq_embedder(input_ids)
        msa_summary = self.msa_embedder(msa_ids, msa_mask, input_ids)
        pair = self.pair_embedder(seq, input_ids)
        sequence_logits = self.sequence_head(seq)
        coords = torch.zeros(
            input_ids.size(0),
            input_ids.size(1),
            self.config.num_atoms,
            3,
            dtype=seq.dtype,
            device=seq.device,
        )

        for recycle_index in range(self.recycle_iters):
            if recycle_index:
                seq, pair = self.recycling(seq, pair, coords)
            seq = seq + msa_summary
            for block_index, block in enumerate(self.e2eformer):
                seq, pair = block(seq, pair, padding_mask)
                if block_index == 0:
                    sequence_logits = self.sequence_head(seq)
                    seq = self.seq_embedder.inject_predicted_bases(seq, sequence_logits, input_ids)
            coords = self.structure_module(seq, pair, padding_mask)

        pair = 0.5 * (pair + pair.transpose(1, 2))
        if not return_aux:
            return coords
        plddt = torch.sigmoid(self.plddt_head(seq)).squeeze(-1) * 100.0
        plddt = plddt.masked_fill(padding_mask, 0.0)
        return {
            "coords": coords,
            "pair_distance_logits": self.distogram_head(pair),
            "plddt": plddt,
            "sequence_logits": sequence_logits,
            "sequence_embedding": seq,
        }

    def combine_task_losses(
        self,
        structure_loss: torch.Tensor,
        sequence_loss: torch.Tensor,
    ) -> torch.Tensor:
        log_variances = self.task_log_variances.clamp(min=-5.0, max=5.0)
        precisions = torch.exp(-log_variances)
        return (
            precisions[0] * structure_loss
            + log_variances[0]
            + precisions[1] * sequence_loss
            + log_variances[1]
        )

    def learned_task_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.exp(-self.task_log_variances.clamp(min=-5.0, max=5.0))
        return weights[0], weights[1]


class SequenceEmbedder(nn.Module):
    def __init__(self, config: RhoFoldConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=RNA3D_PAD_ID)
        self.position = SinusoidalPositionEncoding(d_model=config.d_model, max_len=config.max_len)
        self.norm = nn.LayerNorm(config.d_model)
        self.generated_base_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.generated_norm = nn.LayerNorm(config.d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.norm(self.position(self.embedding(input_ids)))

    def inject_predicted_bases(
        self,
        seq: torch.Tensor,
        sequence_logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        generated_mask = input_ids.eq(RNA3D_MASK_ID)
        if not generated_mask.any():
            return seq
        base_probabilities = F.softmax(sequence_logits[..., 1:5], dim=-1)
        expected_base_embedding = base_probabilities @ self.embedding.weight[1:5]
        generated = self.generated_norm(seq + self.generated_base_projection(expected_base_embedding))
        return torch.where(generated_mask.unsqueeze(-1), generated, seq)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MSAEmbedder(nn.Module):
    def __init__(self, config: RhoFoldConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.msa_dim, padding_idx=RNA3D_PAD_ID)
        self.proj = nn.Linear(config.msa_dim, config.d_model)
        self.fallback = nn.Embedding(config.vocab_size, config.d_model, padding_idx=RNA3D_PAD_ID)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        msa_ids: torch.Tensor | None,
        msa_mask: torch.Tensor | None,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        if msa_ids is None:
            return self.norm(self.fallback(input_ids))
        msa = self.embedding(msa_ids)
        if msa_mask is None:
            msa_mask = msa_ids.eq(RNA3D_PAD_ID)
        weights = (~msa_mask).unsqueeze(-1).to(msa.dtype)
        denom = weights.sum(dim=1).clamp(min=1.0)
        pooled = (msa * weights).sum(dim=1) / denom
        return self.norm(self.proj(pooled))


class PairEmbedder(nn.Module):
    def __init__(self, config: RhoFoldConfig) -> None:
        super().__init__()
        self.left = nn.Linear(config.d_model, config.pair_dim)
        self.right = nn.Linear(config.d_model, config.pair_dim)
        self.relpos = nn.Embedding(65, config.pair_dim)
        self.norm = nn.LayerNorm(config.pair_dim)

    def forward(self, seq: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        length = input_ids.size(1)
        positions = torch.arange(length, device=input_ids.device)
        rel = (positions[None, :] - positions[:, None]).clamp(min=-32, max=32) + 32
        pair = self.left(seq).unsqueeze(2) + self.right(seq).unsqueeze(1)
        pair = pair + self.relpos(rel).unsqueeze(0)
        return self.norm(0.5 * (pair + pair.transpose(1, 2)))


class RecyclingEmbedder(nn.Module):
    def __init__(self, config: RhoFoldConfig) -> None:
        super().__init__()
        self.seq_norm = nn.LayerNorm(config.d_model)
        self.pair_norm = nn.LayerNorm(config.pair_dim)
        self.coord_to_seq = nn.Linear(3, config.d_model)
        self.dist_to_pair = nn.Sequential(
            nn.Linear(1, config.pair_dim),
            nn.GELU(),
            nn.Linear(config.pair_dim, config.pair_dim),
        )

    def forward(self, seq: torch.Tensor, pair: torch.Tensor, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        centers = coords.mean(dim=2)
        distances = torch.cdist(centers, centers).unsqueeze(-1)
        seq = seq + self.coord_to_seq(centers.detach())
        pair = pair + self.dist_to_pair(distances.detach())
        return self.seq_norm(seq), self.pair_norm(pair)


class E2EformerBlock(nn.Module):
    def __init__(self, config: RhoFoldConfig) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.sequence_attention = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.pair_to_seq = nn.Linear(config.pair_dim, config.d_model)
        self.seq_to_pair = nn.Sequential(
            nn.Linear(config.d_model, config.pair_dim),
            nn.GELU(),
            nn.Linear(config.pair_dim, config.pair_dim),
        )
        self.triangle_update = nn.Sequential(
            nn.LayerNorm(config.pair_dim),
            nn.Linear(config.pair_dim, config.pair_dim),
            nn.GELU(),
            nn.Linear(config.pair_dim, config.pair_dim),
        )
        self.seq_norm = nn.LayerNorm(config.d_model)
        self.pair_norm = nn.LayerNorm(config.pair_dim)

    def forward(
        self,
        seq: torch.Tensor,
        pair: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_bias = self.pair_to_seq(pair.mean(dim=2))
        seq = self.seq_norm(seq + seq_bias)
        seq = self.sequence_attention(seq, src_key_padding_mask=padding_mask)
        pair_delta = self.seq_to_pair(seq).unsqueeze(2) + self.seq_to_pair(seq).unsqueeze(1)
        triangle = self.triangle_update(pair)
        pair = self.pair_norm(pair + pair_delta + triangle)
        pair = 0.5 * (pair + pair.transpose(1, 2))
        return seq, pair


class StructureModule(nn.Module):
    def __init__(self, config: RhoFoldConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(config.d_model),
                    nn.Linear(config.d_model, config.d_model),
                    nn.GELU(),
                    nn.Linear(config.d_model, config.d_model),
                )
                for _ in range(config.num_structure_layers)
            ]
        )
        self.pair_to_seq = nn.Linear(config.pair_dim, config.d_model)
        self.coord_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.num_atoms * 3),
        )
        self.num_atoms = config.num_atoms

    def forward(
        self,
        seq: torch.Tensor,
        pair: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden = seq + self.pair_to_seq(pair.mean(dim=2))
        for layer in self.layers:
            hidden = hidden + layer(hidden)
        coords = self.coord_head(hidden).view(hidden.size(0), hidden.size(1), self.num_atoms, 3)
        if padding_mask is not None:
            coords = coords.masked_fill(padding_mask[:, :, None, None], 0.0)
        return coords
