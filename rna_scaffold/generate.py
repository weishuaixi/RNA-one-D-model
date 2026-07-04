from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass

import torch

from rna_scaffold.tokenizer import RnaTokenizer
from rna_scaffold.utils import complementarity_rate, gc_fraction, reverse_complement, validate_rna_sequence


_NATURAL_NON_WC_RIGHT_BASES = {
    "A": ("C", "G"),
    "U": ("G", "C"),
    "C": ("A", "U"),
    "G": ("U", "A"),
}


@dataclass(frozen=True)
class ScaffoldResult:
    left_sequence: str
    motif: str
    right_sequence: str
    left_length: int
    right_length: int
    full_sequence: str
    quality_score: float
    motif_preserved: bool
    left_right_complementarity: float


def build_single_best_result(
    motif: str,
    left_sequence: str,
    quality_score: float,
    mutation_rate: float = 0.0,
    rng_seed: int | None = None,
) -> ScaffoldResult:
    """Build the externally returned single-best JSON result.

    The training target encourages the model to generate both sides, but the
    strict production path can generate one side and construct the other side
    from a mostly reverse-complement template. A small mutation_rate adds
    natural stem defects such as wobble-like or mismatch positions.
    """
    motif = motif.upper()
    left_sequence = left_sequence.upper()
    if not validate_rna_sequence(motif):
        raise ValueError("motif must contain only A, U, C, and G.")
    if not validate_rna_sequence(left_sequence):
        raise ValueError("left_sequence must contain only A, U, C, and G.")
    if not 0 <= mutation_rate <= 0.25:
        raise ValueError("mutation_rate must be in [0, 0.25] to preserve mostly complementary stems.")

    right_sequence = _naturalized_right_sequence(
        left_sequence=left_sequence,
        mutation_rate=mutation_rate,
        rng=random.Random(rng_seed),
    )
    return _make_scaffold_result(motif, left_sequence, right_sequence, quality_score)


def build_random_natural_scaffold_result(
    motif: str,
    min_left_length: int = 30,
    max_left_length: int = 120,
    num_candidates: int = 128,
    rng_seed: int | None = None,
) -> ScaffoldResult:
    """Generate a motif-protected one-dimensional scaffold by rule-based sampling.

    This is a lightweight baseline for early experiments before a trained model
    is available: sample variable-length left stems, derive a mostly
    complementary right stem with natural defects, then return the best-scoring
    candidate.
    """
    motif = motif.upper()
    if not validate_rna_sequence(motif):
        raise ValueError("motif must contain only A, U, C, and G.")
    if min_left_length <= 0:
        raise ValueError("min_left_length must be positive.")
    if max_left_length < min_left_length:
        raise ValueError("max_left_length must be greater than or equal to min_left_length.")
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive.")

    rng = random.Random(rng_seed)
    best: ScaffoldResult | None = None
    for _ in range(num_candidates):
        length = rng.randint(min_left_length, max_left_length)
        left_sequence = "".join(rng.choice("AUCG") for _ in range(length))
        mutation_rate = rng.uniform(0.08, 0.2)
        right_sequence = _naturalized_right_sequence(left_sequence, mutation_rate, rng)
        quality_score = _score_scaffold_candidate(left_sequence, right_sequence)
        candidate = _make_scaffold_result(motif, left_sequence, right_sequence, quality_score)
        if best is None or candidate.quality_score > best.quality_score:
            best = candidate

    if best is None:  # pragma: no cover - guarded by num_candidates validation
        raise RuntimeError("No scaffold candidates were generated.")
    return best


def _naturalized_right_sequence(
    left_sequence: str,
    mutation_rate: float,
    rng: random.Random,
) -> str:
    right = list(reverse_complement(left_sequence))
    if mutation_rate <= 0 or not right:
        return "".join(right)

    mutation_count = round(len(left_sequence) * mutation_rate)
    mutation_count = max(1, min(mutation_count, len(left_sequence)))
    mutated_left_positions = rng.sample(range(len(left_sequence)), mutation_count)

    for left_index in mutated_left_positions:
        right_index = len(left_sequence) - 1 - left_index
        left_base = left_sequence[left_index]
        right[right_index] = rng.choice(_NATURAL_NON_WC_RIGHT_BASES[left_base])
    return "".join(right)


def _make_scaffold_result(
    motif: str,
    left_sequence: str,
    right_sequence: str,
    quality_score: float,
) -> ScaffoldResult:
    rate = complementarity_rate(left_sequence, right_sequence)
    full_sequence = f"{left_sequence}{motif}{right_sequence}"
    return ScaffoldResult(
        left_sequence=left_sequence,
        motif=motif,
        right_sequence=right_sequence,
        left_length=len(left_sequence),
        right_length=len(right_sequence),
        full_sequence=full_sequence,
        quality_score=float(quality_score),
        motif_preserved=full_sequence == f"{left_sequence}{motif}{right_sequence}",
        left_right_complementarity=rate,
    )


def _score_scaffold_candidate(left_sequence: str, right_sequence: str) -> float:
    complementarity = complementarity_rate(left_sequence, right_sequence)
    complementarity_score = max(0.0, 1.0 - abs(complementarity - 0.88) / 0.18)
    gc_score = max(0.0, 1.0 - abs(gc_fraction(left_sequence + right_sequence) - 0.5) / 0.3)
    homopolymer_penalty = max(0, _longest_homopolymer(left_sequence + right_sequence) - 4) * 0.08
    return max(0.0, min(1.0, 0.7 * complementarity_score + 0.3 * gc_score - homopolymer_penalty))


def _longest_homopolymer(sequence: str) -> int:
    longest = 0
    current = 0
    previous = None
    for base in sequence:
        current = current + 1 if base == previous else 1
        previous = base
        longest = max(longest, current)
    return longest


def result_to_json(result: ScaffoldResult) -> str:
    return json.dumps(asdict(result), ensure_ascii=False, indent=2)


@torch.inference_mode()
def greedy_decode_left_seed(
    model,
    tokenizer: RnaTokenizer,
    motif: str,
    max_left_length: int = 128,
    device: str | torch.device = "cpu",
) -> str:
    """Minimal greedy left-side decoder for checkpoints trained with this package.

    This is intentionally conservative: it stops at END_LEFT/EOS/PAD and only
    returns A/U/C/G bases. Production reranking can sit above this function.
    """
    model.eval()
    model.to(device)
    input_ids = torch.tensor([tokenizer.encode(f"<BOS>{motif.upper()}<EOS>")], device=device)
    generated = [tokenizer.bos_token_id, tokenizer.token_to_id["<LEFT>"]]
    for _ in range(max_left_length):
        decoder_input = torch.tensor([generated], device=device)
        logits = model(input_ids=input_ids, decoder_input_ids=decoder_input)
        next_id = int(torch.argmax(logits[0, -1]).item())
        token = tokenizer.id_to_token[next_id]
        if token in {"<END_LEFT>", "<EOS>", "<PAD>", "<RIGHT>"}:
            break
        if token in {"A", "U", "C", "G"}:
            generated.append(next_id)
        else:
            break
    decoded = tokenizer.decode(generated)
    return "".join(base for base in decoded if base in "AUCG")
