from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CandidateMetric:
    motif_id: str
    sequence: str
    valid: bool
    motif_preserved: bool
    total_length: int
    gc_fraction: float
    failure: str | None


@dataclass(frozen=True)
class CandidateSummary:
    count: int
    valid_rate: float
    motif_preservation_rate: float
    unique_rate: float
    mean_length: float
    mean_gc_fraction: float
    failure_count: int


@dataclass(frozen=True)
class BootstrapDifference:
    mean_difference: float
    lower: float
    upper: float
    samples: int
    seed: int


def summarize_candidates(rows: list[CandidateMetric]) -> CandidateSummary:
    if not rows:
        return CandidateSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    count = len(rows)
    return CandidateSummary(
        count=count,
        valid_rate=sum(row.valid for row in rows) / count,
        motif_preservation_rate=sum(row.motif_preserved for row in rows) / count,
        unique_rate=len({row.sequence for row in rows}) / count,
        mean_length=sum(row.total_length for row in rows) / count,
        mean_gc_fraction=sum(row.gc_fraction for row in rows) / count,
        failure_count=sum(row.failure is not None for row in rows),
    )


def normalized_edit_distance(left: str, right: str) -> float:
    if not left and not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1] / max(len(left), len(right))


def kmer_jaccard(left: str, right: str, k: int = 5) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    left_kmers = {left[index : index + k] for index in range(max(0, len(left) - k + 1))}
    right_kmers = {right[index : index + k] for index in range(max(0, len(right) - k + 1))}
    if not left_kmers and not right_kmers:
        return 1.0 if left == right else 0.0
    return len(left_kmers & right_kmers) / len(left_kmers | right_kmers)


def nearest_training_similarity(sequence: str, training_sequences: list[str], k: int = 5) -> float | None:
    if not training_sequences:
        return None
    return max(kmer_jaccard(sequence, reference, k=k) for reference in training_sequences)


def paired_bootstrap(
    left: list[float],
    right: list[float],
    seed: int = 42,
    samples: int = 10000,
) -> BootstrapDifference:
    if len(left) != len(right) or not left:
        raise ValueError("paired inputs must have equal non-zero length")
    if samples <= 0:
        raise ValueError("samples must be positive")
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(differences), size=(samples, len(differences)))
    bootstrapped = differences[indices].mean(axis=1)
    return BootstrapDifference(
        mean_difference=float(differences.mean()),
        lower=float(np.percentile(bootstrapped, 2.5)),
        upper=float(np.percentile(bootstrapped, 97.5)),
        samples=samples,
        seed=seed,
    )
