from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from rna_scaffold.model import ScaffoldModelOutput
from rna_scaffold.tokenizer import RnaTokenizer


@dataclass(frozen=True)
class DecodingSettings:
    denoise_steps: int = 12
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 0.95
    mask_schedule: str = "linear"

    def __post_init__(self) -> None:
        if self.denoise_steps <= 0:
            raise ValueError("denoise_steps must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k is not None and not 1 <= self.top_k <= 4:
            raise ValueError("top_k must be between 1 and 4")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.mask_schedule not in {"linear", "cosine"}:
            raise ValueError("mask_schedule must be 'linear' or 'cosine'")


@dataclass(frozen=True)
class DecodedScaffold:
    sequence: str
    normalized_log_probability: float
    masked_counts: tuple[int, ...]

    @property
    def unresolved_counts(self) -> tuple[int, ...]:
        """Compatibility alias for the V1 decoder field name."""
        return self.masked_counts


def select_flank_lengths(
    output: ScaffoldModelOutput,
    motif_length: int,
    max_length: int,
    generator: torch.Generator,
    sample: bool = True,
    min_scaffold_length: int = 1,
    min_flank_length: int = 0,
) -> tuple[int, int]:
    if output.left_length_logits.shape[0] != 1 or output.right_length_logits.shape[0] != 1:
        raise ValueError("flank-length selection currently requires batch size one")
    if min_scaffold_length < 1 or min_flank_length < 0:
        raise ValueError("invalid minimum scaffold or flank length")
    left_log_probs = output.left_length_logits[0].float().log_softmax(dim=-1)
    right_log_probs = output.right_length_logits[0].float().log_softmax(dim=-1)
    largest_flank = min(
        left_log_probs.shape[-1] - 1,
        right_log_probs.shape[-1] - 1,
        int(max_length) - motif_length - min_flank_length,
    )
    if largest_flank < min_flank_length:
        raise ValueError("motif cannot fit within the requested maximum length")
    lengths = torch.arange(min_flank_length, largest_flank + 1, device=left_log_probs.device)
    pairs = torch.cartesian_prod(lengths, lengths)
    total_lengths = pairs[:, 0] + motif_length + pairs[:, 1]
    legal_pairs = pairs[
        (total_lengths >= min_scaffold_length) & (total_lengths <= int(max_length))
    ]
    if legal_pairs.numel() == 0:
        raise ValueError("motif cannot fit within the requested maximum length")
    pair_scores = left_log_probs[legal_pairs[:, 0]] + right_log_probs[legal_pairs[:, 1]]
    if sample:
        probabilities = torch.softmax(pair_scores, dim=0)
        index = int(torch.multinomial(probabilities, 1, generator=generator).item())
    else:
        index = int(torch.argmax(pair_scores).item())
    return tuple(int(length) for length in legal_pairs[index].tolist())


def _filtered_probabilities(logits: torch.Tensor, settings: DecodingSettings) -> torch.Tensor:
    filtered = logits.float() / settings.temperature
    if settings.top_k is not None and settings.top_k < filtered.shape[-1]:
        threshold = torch.topk(filtered, settings.top_k, dim=-1).values[..., -1:]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
    if settings.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative - sorted_probs >= settings.top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(filtered, float("-inf")).scatter(-1, sorted_indices, sorted_logits)
    return torch.softmax(filtered, dim=-1)


def _scheduled_mask_count(
    initial_masked_count: int,
    step: int,
    settings: DecodingSettings,
) -> int:
    """Return the deterministic number of scaffold tokens to remask after a pass."""
    if step == settings.denoise_steps - 1:
        return 0
    completed_fraction = (step + 1) / settings.denoise_steps
    if settings.mask_schedule == "linear":
        remaining_fraction = 1.0 - completed_fraction
    else:
        remaining_fraction = math.cos(completed_fraction * math.pi / 2)
    return max(0, min(initial_masked_count, math.ceil(initial_masked_count * remaining_fraction)))


@torch.inference_mode()
def iterative_denoise(
    model,
    tokenizer: RnaTokenizer,
    motif: str,
    total_length: int,
    motif_start: int,
    settings: DecodingSettings,
    generator: torch.Generator,
    device: str | torch.device = "cpu",
) -> DecodedScaffold:
    motif = motif.strip().upper().replace("T", "U")
    if len(motif) < 1 or set(motif) - set("AUCG"):
        raise ValueError("motif must contain only A, U, C, and G")
    motif_end = motif_start + len(motif)
    if total_length > int(model.max_length) or motif_start < 0 or motif_end > total_length:
        raise ValueError("invalid motif coordinates for the requested canvas")

    device = torch.device(device)
    mask_id = tokenizer.token_to_id[tokenizer.special.mask]
    canvas = torch.full((1, total_length), mask_id, dtype=torch.long, device=device)
    motif_ids = torch.tensor(tokenizer.encode(motif), dtype=torch.long, device=device)
    canvas[0, motif_start:motif_end] = motif_ids
    fixed = torch.zeros(total_length, dtype=torch.bool, device=device)
    fixed[motif_start:motif_end] = True
    attention = torch.ones_like(canvas, dtype=torch.bool)
    base_token_ids = torch.tensor(
        [tokenizer.token_to_id[base] for base in "AUCG"], dtype=torch.long, device=device
    )
    scaffold_positions = (~fixed).nonzero(as_tuple=False).squeeze(-1)
    initial_masked_count = int(scaffold_positions.numel())
    masked_counts = [initial_masked_count]

    for step in range(settings.denoise_steps):
        output = model(input_ids=canvas, attention_mask=attention)
        probabilities = _filtered_probabilities(output.token_logits[0, scaffold_positions], settings)
        sampled_classes = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
        sampled_probs = probabilities.gather(1, sampled_classes.unsqueeze(-1)).squeeze(-1)
        canvas[0, scaffold_positions] = base_token_ids[sampled_classes]
        next_masked_count = _scheduled_mask_count(initial_masked_count, step, settings)
        if next_masked_count:
            remask_indices = torch.topk(sampled_probs, next_masked_count, largest=False).indices
            canvas[0, scaffold_positions[remask_indices]] = mask_id
        canvas[0, motif_start:motif_end] = motif_ids
        masked_counts.append(next_masked_count)

    if masked_counts[-1] != 0:
        raise RuntimeError("denoising steps ended with unresolved scaffold positions")
    sequence = tokenizer.decode(canvas[0].tolist())
    final_output = model(input_ids=canvas, attention_mask=attention)
    final_probabilities = torch.softmax(final_output.token_logits[0, scaffold_positions].float(), dim=-1)
    final_token_ids = canvas[0, scaffold_positions]
    final_classes = final_token_ids.unsqueeze(-1).eq(base_token_ids).nonzero(as_tuple=False)[:, 1]
    final_log_probs = final_probabilities.gather(1, final_classes.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12).log()
    normalized = float(final_log_probs.mean().item())
    return DecodedScaffold(sequence, normalized, tuple(masked_counts))
