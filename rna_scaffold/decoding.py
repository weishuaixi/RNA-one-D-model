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

    def __post_init__(self) -> None:
        if self.denoise_steps <= 0:
            raise ValueError("denoise_steps must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k is not None and not 1 <= self.top_k <= 4:
            raise ValueError("top_k must be between 1 and 4")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


@dataclass(frozen=True)
class DecodedScaffold:
    sequence: str
    normalized_log_probability: float
    unresolved_counts: tuple[int, ...]


def select_length_position(
    output: ScaffoldModelOutput,
    motif_length: int,
    max_length: int,
    generator: torch.Generator,
    sample: bool = True,
) -> tuple[int, int]:
    if output.length_logits.shape[0] != 1 or output.position_logits.shape[0] != 1:
        raise ValueError("length/position selection currently requires batch size one")
    model_max = min(output.length_logits.shape[-1] - 1, output.position_logits.shape[-1])
    upper = min(int(max_length), model_max)
    pairs: list[tuple[int, int]] = []
    scores: list[torch.Tensor] = []
    length_log_probs = output.length_logits[0].float().log_softmax(dim=-1)
    position_log_probs = output.position_logits[0].float().log_softmax(dim=-1)
    for total_length in range(motif_length + 1, upper + 1):
        for motif_start in range(total_length - motif_length + 1):
            pairs.append((total_length, motif_start))
            scores.append(length_log_probs[total_length] + position_log_probs[motif_start])
    if not pairs:
        raise ValueError("motif cannot fit within the requested maximum length")
    pair_scores = torch.stack(scores)
    if sample:
        probabilities = torch.softmax(pair_scores, dim=0)
        index = int(torch.multinomial(probabilities, 1, generator=generator).item())
    else:
        index = int(torch.argmax(pair_scores).item())
    return pairs[index]


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
    unresolved = ~fixed
    attention = torch.ones_like(canvas, dtype=torch.bool)
    base_token_ids = torch.tensor(
        [tokenizer.token_to_id[base] for base in "AUCG"], dtype=torch.long, device=device
    )
    committed_log_probs = torch.zeros(total_length, dtype=torch.float32, device=device)
    unresolved_counts = [int(unresolved.sum().item())]

    for step in range(settings.denoise_steps):
        if not unresolved.any():
            break
        output = model(input_ids=canvas, attention_mask=attention)
        probabilities = _filtered_probabilities(output.token_logits[0, unresolved], settings)
        sampled_classes = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
        sampled_probs = probabilities.gather(1, sampled_classes.unsqueeze(-1)).squeeze(-1)
        remaining_steps = settings.denoise_steps - step
        commit_count = min(
            int(unresolved.sum().item()),
            max(1, math.ceil(int(unresolved.sum().item()) / remaining_steps)),
        )
        selected = torch.topk(sampled_probs, commit_count).indices
        unresolved_positions = unresolved.nonzero(as_tuple=False).squeeze(-1)
        commit_positions = unresolved_positions[selected]
        canvas[0, commit_positions] = base_token_ids[sampled_classes[selected]]
        committed_log_probs[commit_positions] = sampled_probs[selected].clamp_min(1e-12).log()
        unresolved[commit_positions] = False
        canvas[0, motif_start:motif_end] = motif_ids
        unresolved_counts.append(int(unresolved.sum().item()))

    if unresolved.any():
        raise RuntimeError("denoising steps ended with unresolved scaffold positions")
    sequence = tokenizer.decode(canvas[0].tolist())
    scaffold_mask = ~fixed
    normalized = float(committed_log_probs[scaffold_mask].mean().item())
    return DecodedScaffold(sequence, normalized, tuple(unresolved_counts))
