from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from rna_scaffold.tokenizer import RnaTokenizer
from rna_scaffold.utils import complementarity_rate, validate_rna_sequence

BASES = ("A", "U", "C", "G")


@dataclass
class RnaTrainingPrior:
    lengths: list[int]
    transition: dict[str, dict[str, float]]
    initial: dict[str, float]

    @classmethod
    def empty(cls) -> "RnaTrainingPrior":
        return cls(lengths=[], transition={}, initial={})

    @classmethod
    def from_path(cls, path: str | Path) -> "RnaTrainingPrior":
        from rna_scaffold.data import load_sequences

        return cls.from_sequences(load_sequences(path))

    @classmethod
    def from_sequences(cls, sequences: list[str]) -> "RnaTrainingPrior":
        lengths: list[int] = []
        pair_counts: dict[str, Counter[str]] = {base: Counter() for base in BASES}
        start_counts: Counter[str] = Counter()
        for raw_sequence in sequences:
            sequence = raw_sequence.strip().upper().replace("T", "U")
            if len(sequence) < 2 or not validate_rna_sequence(sequence):
                continue
            lengths.append(len(sequence))
            start_counts[sequence[0]] += 1
            for left, right in zip(sequence, sequence[1:]):
                pair_counts[left][right] += 1

        # Laplace smoothing keeps every base sampleable, including for small
        # or compositionally narrow training files.
        total_start = sum(start_counts.values()) + len(BASES)
        initial = {base: (start_counts.get(base, 0) + 1) / total_start for base in BASES}
        transition: dict[str, dict[str, float]] = {}
        for previous in BASES:
            total = sum(pair_counts[previous].values()) + len(BASES)
            transition[previous] = {
                base: (pair_counts[previous].get(base, 0) + 1) / total for base in BASES
            }

        return cls(lengths=lengths, transition=transition, initial=initial)

    def has_statistics(self) -> bool:
        return bool(self.lengths) and len(self.transition) == len(BASES)

    def sample_total_length(self, motif_length: int, rng: random.Random) -> int | None:
        valid_lengths = [length for length in self.lengths if length > motif_length + 1]
        if not valid_lengths:
            return None
        return rng.choice(valid_lengths)

    def sample_sequence(self, length: int, rng: random.Random) -> str:
        if length <= 0:
            return ""
        if not self.has_statistics():
            return "".join(rng.choice(BASES) for _ in range(length))
        chars = [self._sample_base(None, rng)]
        for _ in range(length - 1):
            chars.append(self._sample_base(chars[-1], rng))
        return "".join(chars)

    def _sample_base(self, previous: str | None, rng: random.Random) -> str:
        probs = self.initial if previous is None else self.transition.get(previous, {})
        if not probs:
            return rng.choice(BASES)
        population, weights = zip(*probs.items())
        return rng.choices(population, weights=weights, k=1)[0]


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


@dataclass(frozen=True)
class MaskedScaffoldPrompt:
    motif: str
    masked_sequence: str
    motif_start: int
    total_length: int


@dataclass(frozen=True)
class GenerationSettings:
    num_candidates: int = 256
    max_length: int = 512
    seed: int = 42
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 0.95
    denoise_steps: int = 12
    max_attempt_multiplier: int = 8
    min_scaffold_length: int = 8
    min_flank_length: int = 2

    def __post_init__(self) -> None:
        if self.num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if self.max_length < 5:
            raise ValueError("max_length must be at least 5")
        if self.max_attempt_multiplier <= 0:
            raise ValueError("max_attempt_multiplier must be positive")
        if self.min_scaffold_length < 1:
            raise ValueError("min_scaffold_length must be positive")
        if self.min_flank_length < 0:
            raise ValueError("min_flank_length must be non-negative")


@dataclass(frozen=True)
class ScaffoldCandidate:
    candidate_id: str
    full_sequence: str
    left_sequence: str
    motif: str
    right_sequence: str
    motif_start: int
    motif_end: int
    total_length: int
    normalized_log_probability: float
    checkpoint_sha256: str
    seed: int
    gc_fraction: float
    max_homopolymer: int
    base_entropy: float
    motif_preserved: bool
    valid: bool
    status: str
    generation_settings: dict = field(default_factory=dict)


def build_auto_masked_scaffold_prompts(
    motif: str,
    num_candidates: int = 16,
    min_total_length: int | None = None,
    max_total_length: int | None = None,
    rng_seed: int | None = None,
) -> list[MaskedScaffoldPrompt]:
    """Create internal mask-inpainting prompts from only a fixed motif.

    The caller supplies the functional motif only. Lengths and motif offsets are
    sampled internally so a downstream masked-scaffold model can infill multiple
    candidate RNA scaffolds and rerank them.
    """
    motif = motif.upper()
    if not validate_rna_sequence(motif):
        raise ValueError("motif must contain only A, U, C, and G.")
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive.")

    default_min = max(len(motif) + 8, len(motif) * 3)
    default_max = max(default_min + 1, len(motif) * 8)
    min_length = default_min if min_total_length is None else min_total_length
    max_length = default_max if max_total_length is None else max_total_length
    if min_length <= len(motif):
        raise ValueError("min_total_length must be greater than motif length.")
    if max_length < min_length:
        raise ValueError("max_total_length must be greater than or equal to min_total_length.")

    rng = random.Random(rng_seed)
    prompts: list[MaskedScaffoldPrompt] = []
    for _ in range(num_candidates):
        total_length = rng.randint(min_length, max_length)
        available_scaffold = total_length - len(motif)
        centered_left = available_scaffold // 2
        jitter_window = max(1, available_scaffold // 4)
        motif_start = min(
            available_scaffold,
            max(0, centered_left + rng.randint(-jitter_window, jitter_window)),
        )
        right_masks = total_length - motif_start - len(motif)
        masked_sequence = "<MASK>" * motif_start + motif + "<MASK>" * right_masks
        prompts.append(
            MaskedScaffoldPrompt(
                motif=motif,
                masked_sequence=masked_sequence,
                motif_start=motif_start,
                total_length=total_length,
            )
        )
    return prompts


def build_motif_scaffold_sequence(
    motif: str,
    num_candidates: int = 128,
    min_total_length: int | None = None,
    max_total_length: int | None = None,
    rng_seed: int | None = None,
    train_data: str | Path | None = None,
) -> ScaffoldResult:
    """Return one complete RNA scaffold sequence from only a fixed motif.

    This is the public motif-only entry point. It mirrors the paper's motif
    scaffolding setup at the interface level: the user supplies a functional
    motif, while masks, candidate lengths, and motif offsets are internal
    generation details.
    """
    motif = motif.upper()
    rng = random.Random(rng_seed)
    prior = RnaTrainingPrior.from_path(train_data) if train_data else RnaTrainingPrior.empty()
    if train_data and min_total_length is None and max_total_length is None and prior.lengths:
        prompts = _build_training_prior_prompts(motif, num_candidates, prior, rng)
    else:
        prompts = build_auto_masked_scaffold_prompts(
            motif=motif,
            num_candidates=num_candidates,
            min_total_length=min_total_length,
            max_total_length=max_total_length,
            rng_seed=rng_seed,
        )
    prompt = rng.choice(prompts)
    left_length = prompt.motif_start
    right_length = prompt.total_length - prompt.motif_start - len(motif)
    left_sequence = prior.sample_sequence(left_length, rng)
    right_sequence = prior.sample_sequence(right_length, rng)
    return _make_scaffold_result(motif, left_sequence, right_sequence, quality_score=0.0)


def generate_markov_baseline(
    motif: str,
    train_data: str | Path | None = None,
    seed: int = 42,
    min_total_length: int | None = None,
    max_total_length: int | None = None,
) -> ScaffoldResult:
    """Explicit first-order Markov baseline; never used as model fallback."""
    return build_motif_scaffold_sequence(
        motif=motif,
        num_candidates=1,
        min_total_length=min_total_length,
        max_total_length=max_total_length,
        rng_seed=seed,
        train_data=train_data,
    )


def _sequence_metrics(sequence: str) -> tuple[float, int, float]:
    counts = Counter(sequence)
    gc_fraction = (counts["G"] + counts["C"]) / len(sequence)
    maximum_run = 1
    current_run = 1
    for previous, current in zip(sequence, sequence[1:]):
        current_run = current_run + 1 if current == previous else 1
        maximum_run = max(maximum_run, current_run)
    entropy = 0.0
    for base in BASES:
        probability = counts[base] / len(sequence)
        if probability:
            entropy -= probability * math.log2(probability)
    return gc_fraction, maximum_run, entropy


def generate_candidates(
    checkpoint: str | Path,
    motif: str,
    settings: GenerationSettings | None = None,
    device: str | torch.device = "cpu",
) -> list[ScaffoldCandidate]:
    """Generate unique motif-preserving candidates with the learned checkpoint."""
    from rna_scaffold.checkpoints import load_scaffold_checkpoint
    from rna_scaffold.decoding import DecodingSettings, iterative_denoise, select_length_position

    motif = motif.strip().upper().replace("T", "U")
    if len(motif) < 4 or not validate_rna_sequence(motif):
        raise ValueError("motif must contain at least four A/U/C/G nucleotides")
    settings = settings or GenerationSettings()
    loaded = load_scaffold_checkpoint(checkpoint, device=device)
    maximum = min(settings.max_length, loaded.max_length)
    required_context = max(settings.min_scaffold_length, 2 * settings.min_flank_length)
    if len(motif) + required_context > maximum:
        raise ValueError("motif cannot fit with the required scaffold context")

    torch_device = torch.device(device)
    generator = torch.Generator(device=torch_device.type).manual_seed(settings.seed)
    motif_input = torch.tensor(
        [loaded.tokenizer.encode(motif)], dtype=torch.long, device=torch_device
    )
    motif_attention = torch.ones_like(motif_input, dtype=torch.bool)
    decoding_settings = DecodingSettings(
        denoise_steps=settings.denoise_steps,
        temperature=settings.temperature,
        top_k=settings.top_k,
        top_p=settings.top_p,
    )
    candidates: list[ScaffoldCandidate] = []
    seen: set[str] = set()
    max_attempts = settings.num_candidates * settings.max_attempt_multiplier
    loaded.model.eval()
    with torch.inference_mode():
        placement_output = loaded.model(motif_input, motif_attention)
        for _ in range(max_attempts):
            if len(candidates) >= settings.num_candidates:
                break
            total_length, motif_start = select_length_position(
                placement_output,
                motif_length=len(motif),
                max_length=maximum,
                generator=generator,
                sample=True,
                min_scaffold_length=settings.min_scaffold_length,
                min_flank_length=settings.min_flank_length,
            )
            decoded = iterative_denoise(
                loaded.model.model,
                loaded.tokenizer,
                motif,
                total_length,
                motif_start,
                decoding_settings,
                generator,
                device=torch_device,
            )
            if decoded.sequence in seen:
                continue
            seen.add(decoded.sequence)
            motif_end = motif_start + len(motif)
            preserved = decoded.sequence[motif_start:motif_end] == motif
            valid = preserved and validate_rna_sequence(decoded.sequence)
            gc_fraction, maximum_run, entropy = _sequence_metrics(decoded.sequence)
            candidates.append(
                ScaffoldCandidate(
                    candidate_id=f"candidate_{len(candidates) + 1:04d}",
                    full_sequence=decoded.sequence,
                    left_sequence=decoded.sequence[:motif_start],
                    motif=motif,
                    right_sequence=decoded.sequence[motif_end:],
                    motif_start=motif_start,
                    motif_end=motif_end,
                    total_length=total_length,
                    normalized_log_probability=decoded.normalized_log_probability,
                    checkpoint_sha256=loaded.checkpoint_sha256,
                    seed=settings.seed,
                    gc_fraction=gc_fraction,
                    max_homopolymer=maximum_run,
                    base_entropy=entropy,
                    motif_preserved=preserved,
                    valid=valid,
                    status="ok" if valid else "invalid",
                    generation_settings=asdict(settings),
                )
            )
    if len(candidates) < settings.num_candidates:
        candidates = [
            candidate
            if candidate.status != "ok"
            else ScaffoldCandidate(**{**asdict(candidate), "status": "shortfall"})
            for candidate in candidates
        ]
    return candidates


def generate_rna_sequence(
    motif: str,
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
    **settings,
) -> str:
    """Return the highest-likelihood sequence from a trained checkpoint."""
    candidates = generate_candidates(
        checkpoint=checkpoint,
        motif=motif,
        settings=GenerationSettings(**settings),
        device=device,
    )
    if not candidates:
        raise RuntimeError("generation produced no valid unique candidates")
    return max(candidates, key=lambda candidate: candidate.normalized_log_probability).full_sequence


def write_candidates_jsonl(candidates: list[ScaffoldCandidate], output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for candidate in candidates:
            handle.write(json.dumps(asdict(candidate), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output_path)


def write_candidates_fasta(candidates: list[ScaffoldCandidate], output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for candidate in candidates:
            handle.write(f">{candidate.candidate_id}\n{candidate.full_sequence}\n")
    temporary.replace(output_path)


def build_single_best_result(
    motif: str,
    left_sequence: str,
    quality_score: float,
    mutation_rate: float = 0.0,
    rng_seed: int | None = None,
) -> ScaffoldResult:
    """Build the externally returned single-best JSON result.

    This compatibility helper keeps the older call shape that supplies a left
    flank, but the right flank is sampled independently as a natural RNA-like
    sequence. No left/right complementarity is imposed or optimized.
    """
    motif = motif.upper()
    left_sequence = left_sequence.upper()
    if not validate_rna_sequence(motif):
        raise ValueError("motif must contain only A, U, C, and G.")
    if not validate_rna_sequence(left_sequence):
        raise ValueError("left_sequence must contain only A, U, C, and G.")
    if not 0 <= mutation_rate <= 0.25:
        raise ValueError("mutation_rate must be in [0, 0.25].")

    rng = random.Random(rng_seed)
    right_sequence = "".join(rng.choice(BASES) for _ in range(len(left_sequence)))
    return _make_scaffold_result(motif, left_sequence, right_sequence, quality_score)


def build_random_natural_scaffold_result(
    motif: str,
    min_left_length: int = 30,
    max_left_length: int = 120,
    num_candidates: int = 128,
    rng_seed: int | None = None,
    train_data: str | Path | None = None,
) -> ScaffoldResult:
    """Generate a motif-protected one-dimensional scaffold by rule-based sampling.

    This is a lightweight baseline for early experiments before a trained model
    is available. Both flanks are sampled independently from the training-data
    prior; no left/right complementarity is imposed.
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
    prior = RnaTrainingPrior.from_path(train_data) if train_data else RnaTrainingPrior.empty()
    length = rng.randint(min_left_length, max_left_length)
    left_sequence = prior.sample_sequence(length, rng)
    right_sequence = prior.sample_sequence(length, rng)
    return _make_scaffold_result(motif, left_sequence, right_sequence, quality_score=0.0)


def _build_training_prior_prompts(
    motif: str,
    num_candidates: int,
    prior: RnaTrainingPrior,
    rng: random.Random,
) -> list[MaskedScaffoldPrompt]:
    prompts: list[MaskedScaffoldPrompt] = []
    for _ in range(num_candidates):
        total_length = prior.sample_total_length(len(motif), rng)
        if total_length is None:
            total_length = max(len(motif) + 8, len(motif) * 3)
        available_scaffold = total_length - len(motif)
        center = available_scaffold // 2
        jitter = max(1, available_scaffold // 4)
        motif_start = min(available_scaffold, max(0, center + rng.randint(-jitter, jitter)))
        right_masks = total_length - motif_start - len(motif)
        prompts.append(
            MaskedScaffoldPrompt(
                motif=motif,
                masked_sequence="<MASK>" * motif_start + motif + "<MASK>" * right_masks,
                motif_start=motif_start,
                total_length=total_length,
            )
        )
    return prompts


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
