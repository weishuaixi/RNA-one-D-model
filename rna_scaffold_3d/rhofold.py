from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from rna_scaffold_3d.geometry import rotation_6d_to_matrix
from rna_scaffold_3d.internal_coords import build_all_atom_from_internal
from rna_scaffold_3d.rna_atoms import (
    RNA_ATOM_TO_INDEX,
    RNA_BASE_ATOMS,
    RNA_NUM_ATOMS,
)
from rna_scaffold_3d.sequence import RNA3D_MASK_ID, RNA3D_PAD_ID


RHO_FOLD_ARCHITECTURE_VERSION = "rna_ipa_internal_coords_v10"


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
    triangle_hidden_dim: int = 32
    triangle_chunk_size: int = 32
    pair_attention_heads: int = 4
    pair_heads: int | None = None
    orientation_bins: int = 24
    equivariant_layers: int = 2
    recycle_stop_gradient: bool = True
    random_recycle_training: bool = True
    activation_checkpointing: bool = False


class RhoFoldModel(nn.Module):
    """Joint RNA sequence/structure model with masked pair geometry and recycling."""

    def __init__(self, config: RhoFoldConfig | None = None, **kwargs) -> None:
        super().__init__()
        self.config = config or RhoFoldConfig(**kwargs)
        config = self.config
        self.num_atoms = config.num_atoms
        self.num_distance_bins = config.num_distance_bins
        self.recycle_iters = max(1, int(config.recycle_iters))
        if config.sequence_loss_initial_weight <= 0:
            raise ValueError("sequence_loss_initial_weight must be positive.")
        if config.d_model % config.nhead:
            raise ValueError("d_model must be divisible by nhead.")
        pair_heads = config.pair_heads or config.pair_attention_heads
        if config.pair_dim % pair_heads:
            raise ValueError("pair_dim must be divisible by pair_heads.")
        self.task_log_variances = nn.Parameter(
            torch.tensor([0.0, -math.log(config.sequence_loss_initial_weight)], dtype=torch.float32)
        )
        self.seq_embedder = SequenceEmbedder(config)
        self.msa_embedder = MSAEmbedder(config)
        self.pair_embedder = PairEmbedder(config)
        self.recycling = RecyclingEmbedder(config)
        self.e2eformer = nn.ModuleList([E2EformerBlock(config) for _ in range(config.num_e2e_layers)])
        self.structure_module = FrameTorsionStructureModule(config)
        self.distogram_head = PairHead(config.pair_dim, config.num_distance_bins)
        self.orientation_heads = nn.ModuleDict(
            {
                "omega": PairHead(config.pair_dim, config.orientation_bins),
                "theta": PairHead(config.pair_dim, config.orientation_bins),
                "phi": PairHead(config.pair_dim, config.orientation_bins // 2),
                "contact": PairHead(config.pair_dim, 1),
            }
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
        recycle_iters: int | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if input_ids.size(1) > self.config.max_len:
            raise ValueError(f"Sequence length {input_ids.size(1)} exceeds max_len={self.config.max_len}.")
        padding_mask = input_ids.eq(RNA3D_PAD_ID) if padding_mask is None else padding_mask.bool()
        residue_mask = ~padding_mask
        pair_mask = residue_mask.unsqueeze(2) & residue_mask.unsqueeze(1)

        initial_seq = self.seq_embedder(input_ids) + self.msa_embedder(msa_ids, msa_mask, input_ids)
        initial_seq = initial_seq * residue_mask.unsqueeze(-1)
        initial_pair = self.pair_embedder(initial_seq, input_ids, pair_mask)
        seq, pair = initial_seq, initial_pair
        coords = torch.zeros(
            input_ids.size(0), input_ids.size(1), self.num_atoms, 3,
            dtype=seq.dtype, device=seq.device,
        )
        sequence_logits = self.sequence_head(seq)
        torsions = torch.zeros(*input_ids.shape, 7, 2, dtype=seq.dtype, device=seq.device)
        sugar_pucker = torch.zeros(*input_ids.shape, 2, dtype=seq.dtype, device=seq.device)
        base_orientation = torch.zeros(
            *input_ids.shape, 3, 3, dtype=seq.dtype, device=seq.device
        )
        frames = torch.zeros(*input_ids.shape, 3, 3, dtype=seq.dtype, device=seq.device)

        iterations = self._resolve_recycle_iterations(recycle_iters)
        for recycle_index in range(iterations):
            final_recycle = recycle_index == iterations - 1
            build_graph = (
                torch.is_grad_enabled()
                and (
                    not self.config.recycle_stop_gradient
                    or final_recycle
                )
            )
            with torch.set_grad_enabled(build_graph):
                if recycle_index:
                    recycled_seq = (
                        seq.detach()
                        if self.config.recycle_stop_gradient
                        else seq
                    )
                    recycled_pair = (
                        pair.detach()
                        if self.config.recycle_stop_gradient
                        else pair
                    )
                    recycled_coords = (
                        coords.detach()
                        if self.config.recycle_stop_gradient
                        else coords
                    )
                    seq_recycle, pair_recycle = self.recycling(
                        recycled_seq,
                        recycled_pair,
                        recycled_coords,
                        pair_mask,
                    )
                    seq = initial_seq + seq_recycle
                    pair = initial_pair + pair_recycle
                seq, pair, sequence_logits, structure = self._fold_iteration(
                    seq,
                    pair,
                    input_ids,
                    padding_mask,
                    pair_mask,
                )
            coords = structure["coords"]
            torsions = structure["torsions"]
            sugar_pucker = structure["sugar_pucker"]
            base_orientation = structure["base_orientation"]
            frames = structure["frames"]

        distance_logits = self.distogram_head(pair)
        distance_logits = 0.5 * (distance_logits + distance_logits.transpose(1, 2))
        distance_logits = distance_logits * pair_mask.unsqueeze(-1)
        if not return_aux:
            return coords
        plddt = torch.sigmoid(self.plddt_head(seq)).squeeze(-1) * 100.0
        plddt = plddt.masked_fill(padding_mask, 0.0)
        orientations = {
            name: head(pair) * pair_mask.unsqueeze(-1)
            for name, head in self.orientation_heads.items()
        }
        return {
            "coords": coords,
            "pair_distance_logits": distance_logits,
            "orientation_logits": orientations,
            "plddt": plddt,
            "sequence_logits": sequence_logits,
            "sequence_embedding": seq,
            "pair_embedding": pair,
            "torsions": torsions,
            "sugar_pucker": sugar_pucker,
            "base_orientation": base_orientation,
            "frames": frames,
            "pair_mask": pair_mask,
        }

    def _fold_iteration(
        self,
        seq: torch.Tensor,
        pair: torch.Tensor,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        for block_index, block in enumerate(self.e2eformer):
            if (
                self.config.activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                seq, pair = checkpoint(
                    block,
                    seq,
                    pair,
                    padding_mask,
                    pair_mask,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                seq, pair = block(seq, pair, padding_mask, pair_mask)
            if block_index == 0:
                injection_logits = self.sequence_head(seq)
                seq = self.seq_embedder.inject_predicted_bases(
                    seq, injection_logits, input_ids
                )
        # The early head above exists only to make masked-base injection
        # available to deeper trunk blocks. Supervision and structure template
        # selection must consume the fully refined representation; otherwise
        # every block after the first is bypassed by the sequence objective.
        sequence_logits = self.sequence_head(seq)
        predicted_base_probabilities = F.softmax(
            sequence_logits[..., 1:5], dim=-1
        )
        known_base = input_ids.ge(1) & input_ids.le(4)
        known_base_probabilities = F.one_hot(
            (input_ids - 1).clamp(min=0, max=3), num_classes=4
        ).to(predicted_base_probabilities.dtype)
        base_probabilities = torch.where(
            known_base.unsqueeze(-1),
            known_base_probabilities,
            predicted_base_probabilities,
        )
        structure = self.structure_module(
            seq,
            pair,
            input_ids,
            padding_mask,
            pair_mask,
            base_probabilities,
        )
        return seq, pair, sequence_logits, structure

    def _resolve_recycle_iterations(self, requested: int | None) -> int:
        if requested is not None:
            return max(1, min(int(requested), self.recycle_iters))
        if self.training and self.config.random_recycle_training and self.recycle_iters > 1:
            return int(torch.randint(1, self.recycle_iters + 1, ()).item())
        return self.recycle_iters

    def _bounded_task_log_variances(self) -> torch.Tensor:
        bounded = self.task_log_variances.clamp(min=-5.0, max=5.0)
        return bounded.detach() + (
            self.task_log_variances - self.task_log_variances.detach()
        )

    def combine_task_losses(self, structure_loss: torch.Tensor, sequence_loss: torch.Tensor) -> torch.Tensor:
        log_variances = self._bounded_task_log_variances()
        precisions = torch.exp(-log_variances)
        return (
            precisions[0] * structure_loss + log_variances[0]
            + precisions[1] * sequence_loss + log_variances[1]
        )

    def learned_task_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.exp(-self._bounded_task_log_variances())
        return weights[0], weights[1]


class PairHead(nn.Sequential):
    def __init__(self, pair_dim: int, output_dim: int) -> None:
        super().__init__(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, pair_dim),
            nn.GELU(),
            nn.Linear(pair_dim, output_dim),
        )


class SequenceEmbedder(nn.Module):
    def __init__(self, config: RhoFoldConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=RNA3D_PAD_ID)
        self.position = SinusoidalPositionEncoding(config.d_model, config.max_len)
        self.norm = nn.LayerNorm(config.d_model)
        self.generated_base_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.generated_norm = nn.LayerNorm(config.d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.norm(self.position(self.embedding(input_ids)))

    def inject_predicted_bases(
        self, seq: torch.Tensor, sequence_logits: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        generated_mask = input_ids.eq(RNA3D_MASK_ID)
        if not generated_mask.any():
            return seq
        probabilities = F.softmax(sequence_logits[..., 1:5], dim=-1)
        expected = probabilities @ self.embedding.weight[1:5]
        generated = self.generated_norm(seq + self.generated_base_projection(expected))
        return torch.where(generated_mask.unsqueeze(-1), generated, seq)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class MSAEmbedder(nn.Module):
    def __init__(self, config: RhoFoldConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.msa_dim, padding_idx=RNA3D_PAD_ID)
        self.proj = nn.Linear(config.msa_dim, config.d_model)
        self.fallback = nn.Embedding(config.vocab_size, config.d_model, padding_idx=RNA3D_PAD_ID)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self, msa_ids: torch.Tensor | None, msa_mask: torch.Tensor | None, input_ids: torch.Tensor
    ) -> torch.Tensor:
        if msa_ids is None:
            return self.norm(self.fallback(input_ids))
        msa = self.embedding(msa_ids)
        msa_mask = msa_ids.eq(RNA3D_PAD_ID) if msa_mask is None else msa_mask.bool()
        weights = (~msa_mask).unsqueeze(-1).to(msa.dtype)
        pooled = (msa * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return self.norm(self.proj(pooled))


class PairEmbedder(nn.Module):
    def __init__(self, config: RhoFoldConfig) -> None:
        super().__init__()
        self.left = nn.Linear(config.d_model, config.pair_dim)
        self.right = nn.Linear(config.d_model, config.pair_dim)
        self.relpos = nn.Embedding(65, config.pair_dim)
        self.contact_channel = nn.Sequential(nn.Linear(config.pair_dim, config.pair_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(config.pair_dim)

    def forward(self, seq: torch.Tensor, input_ids: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        length = input_ids.size(1)
        positions = torch.arange(length, device=input_ids.device)
        relative = (positions[None, :] - positions[:, None]).clamp(-32, 32) + 32
        pair = self.left(seq).unsqueeze(2) + self.right(seq).unsqueeze(1)
        pair = self.norm(pair + self.relpos(relative).unsqueeze(0))
        pair = pair + self.contact_channel(pair)
        return pair * pair_mask.unsqueeze(-1)


class RecyclingEmbedder(nn.Module):
    def __init__(self, config: RhoFoldConfig) -> None:
        super().__init__()
        self.seq_norm = nn.LayerNorm(config.d_model)
        self.pair_norm = nn.LayerNorm(config.pair_dim)
        self.seq_scale = nn.Parameter(torch.tensor(0.01))
        self.pair_scale = nn.Parameter(torch.tensor(0.01))
        self.dist_to_pair = nn.Sequential(
            nn.Linear(1, config.pair_dim), nn.GELU(), nn.Linear(config.pair_dim, config.pair_dim)
        )
        self.pair_to_seq = nn.Linear(config.pair_dim, config.d_model)
        self.representative_atom = RNA_ATOM_TO_INDEX["C1'"]

    def forward(
        self,
        seq: torch.Tensor,
        pair: torch.Tensor,
        coords: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        centers = coords[..., self.representative_atom, :]
        distances = torch.cdist(centers.float(), centers.float()).to(coords.dtype).unsqueeze(-1)
        distance_pair = self.dist_to_pair(distances)
        pair = (
            self.pair_scale * (self.pair_norm(pair) + distance_pair)
        ) * pair_mask.unsqueeze(-1)
        neighbor_count = pair_mask.sum(dim=2, keepdim=True).clamp(min=1).to(pair.dtype)
        pooled_pair = pair.sum(dim=2) / neighbor_count
        seq = (
            self.seq_scale * self.seq_norm(seq)
            + self.pair_to_seq(pooled_pair)
        )
        residue_mask = pair_mask.any(dim=2).unsqueeze(-1)
        return seq * residue_mask, pair


class PairBiasedSelfAttention(nn.Module):
    def __init__(self, config: RhoFoldConfig) -> None:
        super().__init__()
        self.heads = config.nhead
        self.head_dim = config.d_model // config.nhead
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.pair_bias = nn.Linear(config.pair_dim, config.nhead, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, seq: torch.Tensor, pair: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        batch, length, _ = seq.shape
        qkv = self.qkv(seq).view(batch, length, 3, self.heads, self.head_dim)
        q, k, value = (qkv[:, :, index].transpose(1, 2) for index in range(3))
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        logits = logits + self.pair_bias(pair).permute(0, 3, 1, 2)
        logits = logits.masked_fill(~pair_mask.unsqueeze(1), torch.finfo(logits.dtype).min)
        attention = self.dropout(torch.softmax(logits.float(), dim=-1).to(logits.dtype))
        output = torch.matmul(attention, value).transpose(1, 2).reshape(batch, length, -1)
        return self.output(output)


class OuterProductMean(nn.Module):
    """Single-sequence outer-product update (MSA mean degenerates to one row)."""

    def __init__(self, config: RhoFoldConfig) -> None:
        super().__init__()
        hidden = max(8, min(config.triangle_hidden_dim, 32))
        self.hidden = hidden
        self.chunk_size = max(1, config.triangle_chunk_size)
        self.left = nn.Linear(config.d_model, hidden)
        self.right = nn.Linear(config.d_model, hidden)
        self.output = nn.Linear(hidden * hidden, config.pair_dim)

    def forward(self, seq: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        left = self.left(seq)
        right = self.right(seq)
        chunks: list[torch.Tensor] = []
        for start in range(0, seq.size(1), self.chunk_size):
            stop = min(seq.size(1), start + self.chunk_size)
            outer = torch.einsum(
                "bih,bjk->bijhk",
                left[:, start:stop],
                right,
            ).flatten(start_dim=-2)
            chunks.append(self.output(outer))
        update = torch.cat(chunks, dim=1)
        return update * pair_mask.unsqueeze(-1)


class TriangleMultiplicativeUpdate(nn.Module):
    def __init__(self, config: RhoFoldConfig, outgoing: bool) -> None:
        super().__init__()
        hidden = max(8, config.triangle_hidden_dim)
        self.outgoing = outgoing
        self.chunk_size = max(1, config.triangle_chunk_size)
        self.norm = nn.LayerNorm(config.pair_dim)
        self.left = nn.Linear(config.pair_dim, hidden)
        self.right = nn.Linear(config.pair_dim, hidden)
        self.left_gate = nn.Linear(config.pair_dim, hidden)
        self.right_gate = nn.Linear(config.pair_dim, hidden)
        self.output = nn.Linear(hidden, config.pair_dim)
        self.output_gate = nn.Linear(config.pair_dim, config.pair_dim)

    def forward(self, pair: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(pair)
        left = self.left(normalized) * torch.sigmoid(self.left_gate(normalized))
        right = self.right(normalized) * torch.sigmoid(self.right_gate(normalized))
        left = left * pair_mask.unsqueeze(-1)
        right = right * pair_mask.unsqueeze(-1)
        chunks = []
        for start in range(0, pair.size(1), self.chunk_size):
            stop = min(pair.size(1), start + self.chunk_size)
            if self.outgoing:
                chunk = torch.einsum("bikc,bjkc->bijc", left[:, start:stop], right)
            else:
                chunk = torch.einsum("bkic,bkjc->bijc", left[:, :, start:stop], right)
            chunks.append(chunk)
        update = torch.cat(chunks, dim=1)
        valid_residues = pair_mask.any(dim=2).sum(dim=1).clamp(min=1).to(update.dtype)
        update = update / valid_residues.sqrt().view(-1, 1, 1, 1)
        update = self.output(update) * torch.sigmoid(self.output_gate(normalized))
        return update * pair_mask.unsqueeze(-1)


class TriangleAttention(nn.Module):
    def __init__(self, config: RhoFoldConfig, starting: bool) -> None:
        super().__init__()
        self.starting = starting
        self.norm = nn.LayerNorm(config.pair_dim)
        self.heads = config.pair_heads or config.pair_attention_heads
        self.head_dim = config.pair_dim // self.heads
        self.qkv = nn.Linear(config.pair_dim, 3 * config.pair_dim)
        self.triangle_bias = nn.Linear(
            config.pair_dim, self.heads, bias=False
        )
        self.output = nn.Linear(config.pair_dim, config.pair_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.chunk_size = max(1, config.triangle_chunk_size)

    def forward(self, pair: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        if not self.starting:
            pair = pair.transpose(1, 2)
            pair_mask = pair_mask.transpose(1, 2)
        batch, length, _, channels = pair.shape
        normalized_pair = self.norm(pair)
        rows = normalized_pair.reshape(batch * length, length, channels)
        valid_rows = pair_mask.reshape(batch * length, length)
        chunks: list[torch.Tensor] = []
        for start in range(0, rows.size(0), self.chunk_size):
            row = rows[start : start + self.chunk_size]
            valid = valid_rows[start : start + self.chunk_size]
            chunk_batch = row.size(0)
            q, k, value = self.qkv(row).chunk(3, dim=-1)
            reshape = lambda tensor: tensor.view(
                chunk_batch, length, self.heads, self.head_dim
            ).transpose(1, 2)
            q, k, value = reshape(q), reshape(k), reshape(value)
            logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            flat_indices = torch.arange(
                start,
                start + chunk_batch,
                device=pair.device,
            )
            batch_indices = torch.div(
                flat_indices,
                length,
                rounding_mode="floor",
            )
            third_edge_bias = self.triangle_bias(
                normalized_pair[batch_indices]
            ).permute(0, 3, 1, 2)
            logits = logits + third_edge_bias
            logits = logits.masked_fill(
                ~valid[:, None, None, :],
                torch.finfo(logits.dtype).min,
            )
            attention = self.dropout(torch.softmax(logits.float(), dim=-1).to(logits.dtype))
            update = torch.matmul(attention, value).transpose(1, 2).reshape(
                chunk_batch, length, channels
            )
            chunks.append(self.output(update))
        update = torch.cat(chunks, dim=0).reshape(batch, length, length, channels)
        update = update * pair_mask.unsqueeze(-1)
        if not self.starting:
            update = update.transpose(1, 2)
        return update


class PairTransition(nn.Module):
    def __init__(self, config: RhoFoldConfig) -> None:
        super().__init__()
        self.transition = nn.Sequential(
            nn.LayerNorm(config.pair_dim),
            nn.Linear(config.pair_dim, 4 * config.pair_dim),
            nn.GELU(),
            nn.Linear(4 * config.pair_dim, config.pair_dim),
        )

    def forward(self, pair: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        return self.transition(pair) * pair_mask.unsqueeze(-1)


class E2EformerBlock(nn.Module):
    def __init__(self, config: RhoFoldConfig) -> None:
        super().__init__()
        self.seq_norm = nn.LayerNorm(config.d_model)
        self.seq_attention = PairBiasedSelfAttention(config)
        self.seq_transition = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.d_model),
        )
        self.outer_product = OuterProductMean(config)
        self.triangle_out = TriangleMultiplicativeUpdate(config, outgoing=True)
        self.triangle_in = TriangleMultiplicativeUpdate(config, outgoing=False)
        self.triangle_start = TriangleAttention(config, starting=True)
        self.triangle_end = TriangleAttention(config, starting=False)
        self.pair_transition = PairTransition(config)
        self.pair_norm = nn.LayerNorm(config.pair_dim)

    def forward(
        self, seq: torch.Tensor, pair: torch.Tensor,
        padding_mask: torch.Tensor, pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq = seq + self.seq_attention(self.seq_norm(seq), pair, pair_mask)
        seq = seq + self.seq_transition(seq)
        seq = seq.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        pair = pair + self.outer_product(seq, pair_mask)
        pair = pair + self.triangle_out(pair, pair_mask)
        pair = pair + self.triangle_in(pair, pair_mask)
        pair = pair + self.triangle_start(pair, pair_mask)
        pair = pair + self.triangle_end(pair, pair_mask)
        pair = self.pair_norm(pair + self.pair_transition(pair, pair_mask))
        pair = pair * pair_mask.unsqueeze(-1)
        return seq, pair


class PairAttentionPooling(nn.Module):
    def __init__(self, config: RhoFoldConfig) -> None:
        super().__init__()
        self.score = nn.Linear(config.pair_dim, 1)
        self.value = nn.Linear(config.pair_dim, config.d_model)

    def forward(self, pair: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(pair).squeeze(-1)
        logits = logits.masked_fill(~pair_mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits.float(), dim=-1).to(pair.dtype) * pair_mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return torch.einsum("bij,bijd->bid", weights, self.value(pair))


class EquivariantInternalCoordinateRefinement(nn.Module):
    """Refine RNA torsions with SE(3)-invariant messages.

    Updating internal coordinates and rebuilding the chain preserves exact
    covalent geometry. In contrast, independently translating each residue
    after atom construction breaks the O3'-P linkage between residues.
    """

    def __init__(self, config: RhoFoldConfig) -> None:
        super().__init__()
        hidden = max(16, config.pair_dim // 2)
        self.pair_message = nn.Sequential(
            nn.LayerNorm(config.pair_dim),
            nn.Linear(config.pair_dim, hidden),
            nn.SiLU(),
        )
        self.radial_message = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.torsion_delta = nn.Linear(hidden, 7)
        self.max_step_radians = math.radians(20.0)
        nn.init.zeros_(self.torsion_delta.weight)
        nn.init.zeros_(self.torsion_delta.bias)

    def forward(
        self,
        torsions: torch.Tensor,
        origins: torch.Tensor,
        pair: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        relative = origins.unsqueeze(2) - origins.unsqueeze(1)
        distance = torch.linalg.norm(relative, dim=-1, keepdim=True)
        messages = self.pair_message(pair) + self.radial_message(distance)
        messages = messages * pair_mask.unsqueeze(-1)
        neighbor_count = pair_mask.sum(dim=2, keepdim=True).clamp(min=1)
        pooled = messages.sum(dim=2) / neighbor_count.to(messages.dtype)
        delta = self.max_step_radians * torch.tanh(self.torsion_delta(pooled))

        sin_angle, cos_angle = torsions.unbind(dim=-1)
        sin_delta, cos_delta = torch.sin(delta), torch.cos(delta)
        refined = torch.stack(
            (
                sin_angle * cos_delta + cos_angle * sin_delta,
                cos_angle * cos_delta - sin_angle * sin_delta,
            ),
            dim=-1,
        )
        residue_mask = pair_mask.any(dim=-1).unsqueeze(-1).unsqueeze(-1)
        return torch.where(residue_mask, refined, torch.zeros_like(refined))


class InvariantPointAttention(nn.Module):
    """Frame-aware attention invariant to a shared global SE(3) transform."""

    def __init__(self, config: RhoFoldConfig, num_points: int = 2) -> None:
        super().__init__()
        self.heads = config.nhead
        self.head_dim = config.d_model // config.nhead
        self.num_points = num_points
        self.scalar_qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.query_points = nn.Linear(
            config.d_model, self.heads * num_points * 3
        )
        self.key_value_points = nn.Linear(
            config.d_model, self.heads * 2 * num_points * 3
        )
        self.pair_bias = nn.Linear(config.pair_dim, self.heads, bias=False)
        self.pair_value = nn.Linear(
            config.pair_dim, self.heads * self.head_dim, bias=False
        )
        self.point_weights = nn.Parameter(torch.zeros(self.heads))
        output_dim = (
            2 * config.d_model
            + self.heads * self.num_points * 4
        )
        self.output = nn.Linear(output_dim, config.d_model)

    def _to_global(
        self,
        points: torch.Tensor,
        rotations: torch.Tensor,
        origins: torch.Tensor,
    ) -> torch.Tensor:
        return (
            torch.einsum("blij,blhpj->blhpi", rotations, points)
            + origins[:, :, None, None, :]
        )

    def forward(
        self,
        seq: torch.Tensor,
        pair: torch.Tensor,
        rotations: torch.Tensor,
        origins: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = seq.shape
        scalar = self.scalar_qkv(seq).view(
            batch, length, 3, self.heads, self.head_dim
        )
        q, k, value = (scalar[:, :, index] for index in range(3))
        scalar_logits = torch.einsum("bihc,bjhc->bhij", q, k)
        scalar_logits = scalar_logits / math.sqrt(3.0 * self.head_dim)

        query_points = self.query_points(seq).view(
            batch, length, self.heads, self.num_points, 3
        )
        key_value_points = self.key_value_points(seq).view(
            batch, length, self.heads, 2 * self.num_points, 3
        )
        key_points, value_points = key_value_points.split(
            self.num_points, dim=-2
        )
        query_global = self._to_global(query_points, rotations, origins)
        key_global = self._to_global(key_points, rotations, origins)
        value_global = self._to_global(value_points, rotations, origins)
        squared_distance = (
            query_global.unsqueeze(2) - key_global.unsqueeze(1)
        ).square().sum(dim=-1).sum(dim=-1)
        point_scale = F.softplus(self.point_weights).view(1, 1, 1, self.heads)
        point_logits = -0.5 * squared_distance * point_scale
        point_logits = point_logits.permute(0, 3, 1, 2)

        logits = (
            scalar_logits / math.sqrt(3.0)
            + self.pair_bias(pair).permute(0, 3, 1, 2) / math.sqrt(3.0)
            + point_logits / math.sqrt(3.0 * max(1, self.num_points))
        )
        logits = logits.masked_fill(
            ~pair_mask.unsqueeze(1), torch.finfo(logits.dtype).min
        )
        attention = torch.softmax(logits.float(), dim=-1).to(seq.dtype)
        attention = attention * pair_mask.unsqueeze(1)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        scalar_output = torch.einsum(
            "bhij,bjhc->bihc", attention, value
        ).reshape(batch, length, -1)
        point_output_global = torch.einsum(
            "bhij,bjhpc->bihpc", attention, value_global
        )
        point_delta = point_output_global - origins[:, :, None, None, :]
        point_output_local = torch.einsum(
            "blji,blhpj->blhpi", rotations, point_delta
        )
        point_norm = torch.sqrt(
            point_output_local.square().sum(dim=-1) + 1e-8
        )
        pair_values = self.pair_value(pair).view(
            batch, length, length, self.heads, self.head_dim
        )
        pair_output = torch.einsum(
            "bhij,bijhc->bihc", attention, pair_values
        ).reshape(batch, length, -1)
        features = torch.cat(
            (
                scalar_output,
                point_output_local.reshape(batch, length, -1),
                point_norm.reshape(batch, length, -1),
                pair_output,
            ),
            dim=-1,
        )
        residue_mask = pair_mask.any(dim=-1).unsqueeze(-1)
        return self.output(features) * residue_mask


def _canonical_rna_templates(num_atoms: int) -> torch.Tensor:
    """A/U/C/G base vectors from wwPDB CCD ideal coordinates.

    Coordinates are expressed relative to C1' in the canonical
    C4'->C3'/C5' residue frame. Missing atom slots remain at C1' (zero
    displacement), so probability-weighted templates stay differentiable.
    """
    # Source: wwPDB Chemical Component Dictionary entries A/U/C/G,
    # pdbx_model_Cartn_*_ideal, accessed 2026-07-30.
    base_values = {
        "A": {
            "N1": (0.9267, -3.9673, -3.2221), "C2": (0.9767, -3.9457, -1.9033),
            "N3": (0.7163, -2.8625, -1.2028), "C4": (0.3872, -1.7272, -1.8090),
            "C5": (0.3255, -1.6960, -3.2125), "C6": (0.6060, -2.8803, -3.9149),
            "N9": (0.0652, -0.4616, -1.3880), "C8": (-0.1819, 0.2876, -2.4993),
            "N7": (-0.0320, -0.4393, -3.5679), "N6": (0.5556, -2.9115, -5.2976),
        },
        "U": {
            "N1": (0.0684, -0.4584, -1.3903), "C2": (0.3626, 0.4170, -2.3661),
            "O2": (0.5737, 1.5800, -2.0848), "N3": (0.4370, 0.0210, -3.6512),
            "C4": (0.2050, -1.2660, -3.9800), "C5": (-0.1107, -2.1961, -2.9619),
            "C6": (-0.1731, -1.7734, -1.6810), "O4": (0.2671, -1.6233, -5.1427),
        },
        "C": {
            "N1": (0.0665, -0.4603, -1.3892), "C2": (0.3990, 0.4032, -2.3640),
            "O2": (0.6428, 1.5645, -2.0805), "N3": (0.4669, 0.0124, -3.6360),
            "C4": (0.2062, -1.2439, -3.9724), "N4": (0.2787, -1.6368, -5.2882),
            "C5": (-0.1503, -2.1730, -2.9737), "C6": (-0.2084, -1.7614, -1.6854),
        },
        "G": {
            "N1": (0.9414, -3.9880, -3.2213), "C2": (0.9821, -3.9538, -1.8601),
            "N2": (1.3160, -5.0952, -1.1751), "N3": (0.7171, -2.8590, -1.1842),
            "C4": (0.3871, -1.7232, -1.8095), "C5": (0.3215, -1.6958, -3.2064),
            "C6": (0.6092, -2.8818, -3.9214), "N9": (0.0658, -0.4615, -1.3886),
            "C8": (-0.1818, 0.2904, -2.5000), "N7": (-0.0308, -0.4376, -3.5680),
            "O6": (0.5637, -2.9028, -5.1397),
        },
    }
    # Express the CCD base plane in the sugar frame used by the internal
    # coordinate builder.  The raw CCD-derived vectors preserve base
    # geometry, but their original frame leaves the glycosidic axis at
    # nonphysical O4'-C1'-N and C2'-C1'-N angles when combined with our
    # canonical ribose.  A single rigid reorientation preserves every base
    # bond/angle while matching the observed RNA glycosidic geometry.
    target_glycosidic_direction = F.normalize(
        torch.tensor([0.3182557, 0.2362722, -0.9180897]), dim=0
    )
    templates = []
    for base in ("A", "U", "C", "G"):
        coords = torch.zeros(num_atoms, 3)
        for name, value in base_values[base].items():
            coords[RNA_ATOM_TO_INDEX[name]] = torch.tensor(value)
        glycosidic_atom = "N9" if base in ("A", "G") else "N1"
        source = F.normalize(
            coords[RNA_ATOM_TO_INDEX[glycosidic_atom]], dim=0
        )
        cross = torch.cross(source, target_glycosidic_direction, dim=0)
        cosine = torch.dot(source, target_glycosidic_direction)
        skew = torch.stack(
            (
                torch.stack((cross.new_zeros(()), -cross[2], cross[1])),
                torch.stack((cross[2], cross.new_zeros(()), -cross[0])),
                torch.stack((-cross[1], cross[0], cross.new_zeros(()))),
            )
        )
        rotation = (
            torch.eye(3)
            + skew
            + skew @ skew / (1.0 + cosine).clamp(min=1e-6)
        )
        coords = coords @ rotation.T
        templates.append(coords)
    return torch.stack(templates)


def _canonical_rna_template(num_atoms: int) -> torch.Tensor:
    """Backward-compatible A template accessor."""
    return _canonical_rna_templates(num_atoms)[0]


class FrameTorsionStructureModule(nn.Module):
    """Build all atoms from residue rigid frames, seven RNA torsions and a template."""

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
        self.pair_pool = PairAttentionPooling(config)
        self.pair_attention = nn.ModuleList(
            [PairBiasedSelfAttention(config) for _ in range(config.num_structure_layers)]
        )
        self.pair_attention_norm = nn.ModuleList(
            [nn.LayerNorm(config.d_model) for _ in range(config.num_structure_layers)]
        )
        self.ipa_norm = nn.ModuleList(
            [nn.LayerNorm(config.d_model) for _ in range(config.num_structure_layers)]
        )
        self.ipa = nn.ModuleList(
            [InvariantPointAttention(config) for _ in range(config.num_structure_layers)]
        )
        self.rotation_head = nn.Linear(config.d_model, 6)
        self.torsion_head = nn.Linear(config.d_model, 7 * 2)
        self.sugar_pucker_head = nn.Linear(config.d_model, 2)
        self.base_orientation_head = nn.Linear(config.d_model, 6)
        self.refiners = nn.ModuleList(
            [
                EquivariantInternalCoordinateRefinement(config)
                for _ in range(config.equivariant_layers)
            ]
        )
        self.num_atoms = config.num_atoms
        self.register_buffer(
            "atom_templates", _canonical_rna_templates(config.num_atoms)
        )
        self._initialize_physical_geometry_heads()

    def _initialize_physical_geometry_heads(self) -> None:
        """Start from an A-form-like RNA chain rather than random torsions."""
        with torch.no_grad():
            # Small non-zero weights retain the physical bias while ensuring
            # that the first structure loss already reaches the trunk. Exact
            # zero initialization makes every initial coordinate independent
            # of sequence features and renders recycling inert until an
            # optimizer step changes the geometry heads.
            nn.init.normal_(self.rotation_head.weight, std=1e-3)
            self.rotation_head.bias.copy_(
                torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            )
            nn.init.normal_(self.torsion_head.weight, std=1e-3)
            a_form_degrees = (-62.0, 180.0, 54.0, 82.0, -152.0, -71.0, -160.0)
            torsion_bias = []
            for degrees in a_form_degrees:
                radians = math.radians(degrees)
                torsion_bias.extend((math.sin(radians), math.cos(radians)))
            self.torsion_head.bias.copy_(torch.tensor(torsion_bias))
            nn.init.normal_(self.sugar_pucker_head.weight, std=1e-3)
            c3_endo_phase = math.radians(18.0)
            self.sugar_pucker_head.bias.copy_(
                torch.tensor([math.sin(c3_endo_phase), math.cos(c3_endo_phase)])
            )
            nn.init.normal_(self.base_orientation_head.weight, std=1e-3)
            self.base_orientation_head.bias.copy_(
                torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            )

    def forward(
        self, seq: torch.Tensor, pair: torch.Tensor, input_ids: torch.Tensor,
        padding_mask: torch.Tensor, pair_mask: torch.Tensor,
        base_probabilities: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = seq + self.pair_pool(pair, pair_mask)
        for attention_norm, attention, ipa_norm, ipa, layer in zip(
            self.pair_attention_norm,
            self.pair_attention,
            self.ipa_norm,
            self.ipa,
            self.layers,
        ):
            hidden = hidden + attention(attention_norm(hidden), pair, pair_mask)
            provisional = self._predict_geometry(
                hidden, padding_mask, base_probabilities
            )
            hidden = hidden + ipa(
                ipa_norm(hidden),
                pair,
                provisional["frames"],
                provisional["origins"],
                pair_mask,
            )
            hidden = hidden + layer(hidden)
            hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        geometry = self._predict_geometry(
            hidden, padding_mask, base_probabilities
        )
        for refiner in self.refiners:
            refined_torsions = refiner(
                geometry["torsions"],
                geometry["origins"],
                pair,
                pair_mask,
            )
            geometry = self._predict_geometry(
                hidden,
                padding_mask,
                base_probabilities,
                torsions_override=refined_torsions,
            )
        return {
            "coords": geometry["coords"],
            "torsions": geometry["torsions"],
            "sugar_pucker": geometry["sugar_pucker"],
            "base_orientation": geometry["base_orientation"],
            "frames": geometry["frames"],
            "origins": geometry["origins"],
        }

    def _predict_geometry(
        self,
        hidden: torch.Tensor,
        padding_mask: torch.Tensor,
        base_probabilities: torch.Tensor,
        torsions_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # NeRF is a long sequential coordinate recurrence. Keep it and all
        # frame/torsion geometry in FP32 even when the trunk runs under AMP.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            geometry_hidden = hidden.float()
            initial_rotation = rotation_6d_to_matrix(
                self.rotation_head(geometry_hidden[:, 0])
            )
            torsions = F.normalize(
                self.torsion_head(geometry_hidden).view(
                    *geometry_hidden.shape[:2], 7, 2
                ),
                dim=-1,
                eps=1e-6,
            )
            if torsions_override is not None:
                if torsions_override.shape != torsions.shape:
                    raise ValueError(
                        "torsions_override must match the predicted torsion shape."
                    )
                torsions = F.normalize(
                    torsions_override.float(), dim=-1, eps=1e-6
                )
            sugar_pucker = F.normalize(
                self.sugar_pucker_head(geometry_hidden), dim=-1, eps=1e-6
            )
            base_orientation = rotation_6d_to_matrix(
                self.base_orientation_head(geometry_hidden)
            )
            residue_templates = torch.einsum(
                "blt,taj->blaj",
                base_probabilities.float(),
                self.atom_templates.float(),
            )
            c1 = RNA_ATOM_TO_INDEX["C1'"]
            glycosidic_indices = torch.tensor(
                [
                    RNA_ATOM_TO_INDEX["N9"],
                    RNA_ATOM_TO_INDEX["N1"],
                    RNA_ATOM_TO_INDEX["N1"],
                    RNA_ATOM_TO_INDEX["N9"],
                ],
                device=geometry_hidden.device,
            )
            chi_reference_indices = torch.tensor(
                [
                    RNA_ATOM_TO_INDEX["C4"],
                    RNA_ATOM_TO_INDEX["C2"],
                    RNA_ATOM_TO_INDEX["C2"],
                    RNA_ATOM_TO_INDEX["C4"],
                ],
                device=geometry_hidden.device,
            )
            base_index = torch.arange(4, device=geometry_hidden.device)
            glycosidic_vectors = torch.einsum(
                "blt,tj->blj",
                base_probabilities.float(),
                self.atom_templates[base_index, glycosidic_indices].float()
                - self.atom_templates[:, c1].float(),
            )
            chi_reference_vectors = torch.einsum(
                "blt,tj->blj",
                base_probabilities.float(),
                self.atom_templates[base_index, chi_reference_indices].float()
                - self.atom_templates[:, c1].float(),
            )
            coords, rotations, origins = build_all_atom_from_internal(
                torsions=torsions,
                sugar_pucker=sugar_pucker,
                base_orientation=base_orientation,
                padding_mask=padding_mask,
                initial_rotation=initial_rotation,
                atom_template=residue_templates,
                glycosidic_vector=glycosidic_vectors,
                chi_reference_vector=chi_reference_vectors,
            )
        return {
            "coords": coords,
            "torsions": torsions,
            "sugar_pucker": sugar_pucker,
            "base_orientation": base_orientation,
            "frames": rotations,
            "origins": origins,
        }


# Kept as an import-compatible name for older callers.
StructureModule = FrameTorsionStructureModule
