from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path

import torch

from rna_scaffold.geometry import validate_scaffold_length_limit
from rna_scaffold.utils import validate_rna_sequence

BASES = ("A", "U", "C", "G")


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
    max_attempts: int | None = None
    min_normalized_edit_distance: float = 0.0
    max_kmer_similarity: float | None = None
    kmer_size: int = 5
    max_homopolymer_run: int | None = None
    gc_min: float | None = None
    gc_max: float | None = None
    enforce_gc_bounds: bool = False
    min_scaffold_length: int = 8
    min_flank_length: int = 2

    def __post_init__(self) -> None:
        if self.num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if self.max_length < 5:
            raise ValueError("max_length must be at least 5")
        validate_scaffold_length_limit(self.max_length)
        if self.max_attempt_multiplier <= 0:
            raise ValueError("max_attempt_multiplier must be positive")
        if self.max_attempts is not None and self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if not 0 <= self.min_normalized_edit_distance <= 1:
            raise ValueError("min_normalized_edit_distance must be in [0, 1]")
        if self.max_kmer_similarity is not None and not 0 <= self.max_kmer_similarity <= 1:
            raise ValueError("max_kmer_similarity must be in [0, 1]")
        if self.kmer_size <= 0:
            raise ValueError("kmer_size must be positive")
        if self.max_homopolymer_run is not None and self.max_homopolymer_run <= 0:
            raise ValueError("max_homopolymer_run must be positive")
        if self.gc_min is not None and not 0 <= self.gc_min <= 1:
            raise ValueError("gc_min must be in [0, 1]")
        if self.gc_max is not None and not 0 <= self.gc_max <= 1:
            raise ValueError("gc_max must be in [0, 1]")
        if self.gc_min is not None and self.gc_max is not None and self.gc_min > self.gc_max:
            raise ValueError("gc_min must not exceed gc_max")
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
    architecture_version: int = 2
    generation_audit: dict = field(default_factory=dict)
    gc_in_soft_range: bool = True
    gc_soft_penalty: float = 0.0


@dataclass(frozen=True)
class CandidateDiversityDecision:
    accepted: bool
    reason: str
    metadata: dict[str, float | bool]


@dataclass(frozen=True)
class CandidateGenerationAudit:
    requested: int
    max_attempts: int
    attempted: int
    accepted: int
    rejected: dict[str, int]


class GeneratedCandidates(list[ScaffoldCandidate]):
    """A list-compatible generated batch with reproducibility metadata."""

    def __init__(
        self,
        candidates: Sequence[ScaffoldCandidate] = (),
        audit: CandidateGenerationAudit | None = None,
        generation_settings: dict | None = None,
    ) -> None:
        super().__init__(candidates)
        self.audit = audit or CandidateGenerationAudit(
            requested=len(self), max_attempts=len(self), attempted=len(self), accepted=len(self), rejected={}
        )
        self.generation_settings = dict(
            generation_settings if generation_settings is not None else (self[0].generation_settings if self else {})
        )


def _sequence_metrics(sequence: str) -> tuple[float, int, float]:
    counts = Counter(sequence)
    gc_fraction = (counts["G"] + counts["C"]) / len(sequence)
    maximum_run = 1
    current_run = 1
    for previous, current in pairwise(sequence):
        current_run = current_run + 1 if current == previous else 1
        maximum_run = max(maximum_run, current_run)
    entropy = 0.0
    for base in BASES:
        probability = counts[base] / len(sequence)
        if probability:
            entropy -= probability * math.log2(probability)
    return gc_fraction, maximum_run, entropy


def _normalized_edit_distance(left: str, right: str) -> float:
    """Levenshtein distance scaled by the longer candidate length."""
    if not left and not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for left_index, left_base in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_base in enumerate(right, start=1):
            current.append(min(
                previous[right_index] + 1,
                current[right_index - 1] + 1,
                previous[right_index - 1] + (left_base != right_base),
            ))
        previous = current
    return previous[-1] / max(len(left), len(right))


def _kmer_similarity(left: str, right: str, k: int) -> float:
    left_kmers = {left[index:index + k] for index in range(max(0, len(left) - k + 1))}
    right_kmers = {right[index:index + k] for index in range(max(0, len(right) - k + 1))}
    if not left_kmers and not right_kmers:
        return 1.0
    return len(left_kmers & right_kmers) / len(left_kmers | right_kmers)


def _gc_metadata(gc_fraction: float, settings: GenerationSettings) -> dict[str, float | bool]:
    if settings.gc_min is None and settings.gc_max is None:
        return {}
    lower_distance = max(0.0, (settings.gc_min or 0.0) - gc_fraction)
    upper_distance = max(0.0, gc_fraction - (settings.gc_max if settings.gc_max is not None else 1.0))
    return {"gc_in_soft_range": not (lower_distance or upper_distance), "gc_soft_penalty": lower_distance + upper_distance}


def assess_candidate_diversity(
    sequence: str,
    accepted_sequences: Sequence[str],
    settings: GenerationSettings,
) -> CandidateDiversityDecision:
    """Pure, ordered diversity policy for one proposed RNA sequence."""
    if sequence in accepted_sequences:
        return CandidateDiversityDecision(False, "duplicate", {})
    if any(_normalized_edit_distance(sequence, accepted) < settings.min_normalized_edit_distance for accepted in accepted_sequences):
        return CandidateDiversityDecision(False, "near_duplicate", {})
    if settings.max_kmer_similarity is not None and any(
        _kmer_similarity(sequence, accepted, settings.kmer_size) > settings.max_kmer_similarity
        for accepted in accepted_sequences
    ):
        return CandidateDiversityDecision(False, "kmer_similarity", {})
    gc_fraction, maximum_run, _ = _sequence_metrics(sequence)
    if settings.max_homopolymer_run is not None and maximum_run > settings.max_homopolymer_run:
        return CandidateDiversityDecision(False, "homopolymer", {})
    metadata = _gc_metadata(gc_fraction, settings)
    if settings.enforce_gc_bounds and metadata and not metadata["gc_in_soft_range"]:
        return CandidateDiversityDecision(False, "gc_out_of_bounds", {})
    return CandidateDiversityDecision(True, "accepted", metadata)


def generate_candidates(
    checkpoint: str | Path,
    motif: str,
    settings: GenerationSettings | None = None,
    device: str | torch.device = "cpu",
    loaded_checkpoint: object | None = None,
) -> GeneratedCandidates:
    """Generate unique motif-preserving candidates with the learned checkpoint."""
    from rna_scaffold.checkpoints import load_scaffold_checkpoint
    from rna_scaffold.decoding import (
        DecodingSettings,
        build_flank_pair_sampler,
        iterative_denoise,
    )

    motif = motif.strip().upper().replace("T", "U")
    if len(motif) < 4 or not validate_rna_sequence(motif):
        raise ValueError("motif must contain at least four A/U/C/G nucleotides")
    settings = settings or GenerationSettings()
    loaded = (
        loaded_checkpoint
        if loaded_checkpoint is not None
        else load_scaffold_checkpoint(checkpoint, device=device)
    )
    maximum = min(settings.max_length, loaded.max_length)
    minimum_total_length = max(settings.min_scaffold_length, len(motif) + 2 * settings.min_flank_length)
    if minimum_total_length > maximum:
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
    max_attempts = settings.max_attempts or settings.num_candidates * settings.max_attempt_multiplier
    rejected: Counter[str] = Counter()
    attempts = 0
    loaded.model.eval()
    with torch.inference_mode():
        placement_output = loaded.model(motif_input, motif_attention)
        pair_sampler = build_flank_pair_sampler(
            placement_output,
            motif_length=len(motif),
            max_length=maximum,
            min_scaffold_length=settings.min_scaffold_length,
            min_flank_length=settings.min_flank_length,
        )
        for _ in range(max_attempts):
            if len(candidates) >= settings.num_candidates:
                break
            attempts += 1
            left_length, right_length = pair_sampler.select(generator, sample=True)
            total_length = left_length + len(motif) + right_length
            motif_start = left_length
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
            decision = assess_candidate_diversity(decoded.sequence, [candidate.full_sequence for candidate in candidates], settings)
            if not decision.accepted:
                rejected[decision.reason] += 1
                continue
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
                    gc_in_soft_range=bool(decision.metadata.get("gc_in_soft_range", True)),
                    gc_soft_penalty=float(decision.metadata.get("gc_soft_penalty", 0.0)),
                )
            )
    audit = CandidateGenerationAudit(
        requested=settings.num_candidates,
        max_attempts=max_attempts,
        attempted=attempts,
        accepted=len(candidates),
        rejected=dict(sorted(rejected.items())),
    )
    status = "ok" if len(candidates) >= settings.num_candidates else "shortfall"
    candidates = [
        ScaffoldCandidate(
            **{
                **asdict(candidate),
                "status": candidate.status if candidate.status != "ok" else status,
                "generation_audit": asdict(audit),
            }
        )
        for candidate in candidates
    ]
    candidates.sort(
        key=lambda candidate: (candidate.gc_soft_penalty, -candidate.normalized_log_probability, candidate.candidate_id)
    )
    return GeneratedCandidates(candidates, audit, asdict(settings))


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


def _artifact_metadata(candidates: Sequence[ScaffoldCandidate]) -> dict:
    if isinstance(candidates, GeneratedCandidates):
        return {
            "architecture_version": 2,
            "generation_settings": candidates.generation_settings,
            "generation_audit": asdict(candidates.audit),
        }
    if candidates:
        first = candidates[0]
        audit = first.generation_audit or asdict(
            CandidateGenerationAudit(len(candidates), len(candidates), len(candidates), len(candidates), {})
        )
        return {
            "architecture_version": first.architecture_version,
            "generation_settings": first.generation_settings,
            "generation_audit": audit,
        }
    return {
        "architecture_version": 2,
        "generation_settings": {},
        "generation_audit": asdict(CandidateGenerationAudit(0, 0, 0, 0, {})),
    }


def write_candidates_jsonl(candidates: Sequence[ScaffoldCandidate], output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    metadata = _artifact_metadata(candidates)
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        if not candidates:
            handle.write(
                json.dumps(
                    {
                        "record_type": "generation_metadata",
                        **metadata,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
        for candidate in candidates:
            row = asdict(candidate)
            row.update(metadata)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output_path)


def write_candidates_fasta(candidates: Sequence[ScaffoldCandidate], output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    metadata = _artifact_metadata(candidates)
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write('; ' + json.dumps({"architecture_version": metadata["architecture_version"]}, sort_keys=True) + "\n")
        handle.write('; ' + json.dumps({"generation_settings": metadata["generation_settings"]}, sort_keys=True) + "\n")
        handle.write('; ' + json.dumps({"generation_audit": metadata["generation_audit"]}, sort_keys=True) + "\n")
        for candidate in candidates:
            handle.write(f">{candidate.candidate_id}\n{candidate.full_sequence}\n")
    temporary.replace(output_path)
