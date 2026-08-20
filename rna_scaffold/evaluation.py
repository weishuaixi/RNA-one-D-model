from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

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
    normalized_edit_diversity: float = 0.0
    kmer_diversity: float = 0.0
    nearest_training_kmer_similarity: float | None = None
    max_homopolymer_run: int = 0
    runtime_seconds: float = 0.0
    peak_cuda_memory_bytes: int | None = None
    rejection_reason: str | None = None
    rnafold_status: str | None = None
    rnafold_dot_bracket: str | None = None
    rnafold_mfe_kcal_mol: float | None = None
    rnafold_paired_fraction: float | None = None
    rnafold_motif_paired_fraction: float | None = None
    rnafold_runtime_seconds: float | None = None
    rnafold_version: str | None = None
    rnafold_error: str | None = None


@dataclass(frozen=True)
class CandidateSummary:
    count: int
    valid_rate: float
    motif_preservation_rate: float
    unique_rate: float
    mean_length: float
    mean_gc_fraction: float
    failure_count: int
    mean_normalized_edit_diversity: float
    mean_kmer_diversity: float
    mean_nearest_training_kmer_similarity: float | None
    mean_max_homopolymer_run: float
    maximum_homopolymer_run: int
    runtime_seconds: float
    peak_cuda_memory_bytes: int | None
    rnafold_ok_rate: float | None
    mean_rnafold_mfe_kcal_mol: float | None
    mean_rnafold_paired_fraction: float | None
    mean_rnafold_motif_paired_fraction: float | None
    rnafold_failure_count: int


@dataclass(frozen=True)
class BootstrapDifference:
    mean_difference: float
    lower: float
    upper: float
    samples: int
    seed: int


def summarize_candidates(rows: list[CandidateMetric]) -> CandidateSummary:
    if not rows:
        return CandidateSummary(
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            0.0,
            0.0,
            None,
            0.0,
            0,
            0.0,
            None,
            None,
            None,
            None,
            None,
            0,
        )
    count = len(rows)
    nearest_training = [
        row.nearest_training_kmer_similarity for row in rows if row.nearest_training_kmer_similarity is not None
    ]
    peak_memory = [row.peak_cuda_memory_bytes for row in rows if row.peak_cuda_memory_bytes is not None]
    rnafold_rows = [row for row in rows if row.rnafold_status is not None]
    sequence_rows = [row for row in rows if row.sequence]
    return CandidateSummary(
        count=count,
        valid_rate=sum(row.valid for row in rows) / count,
        motif_preservation_rate=sum(row.motif_preserved for row in rows) / count,
        unique_rate=len({row.sequence for row in sequence_rows}) / count,
        mean_length=(
            sum(row.total_length for row in sequence_rows) / len(sequence_rows) if sequence_rows else 0.0
        ),
        mean_gc_fraction=(
            sum(row.gc_fraction for row in sequence_rows) / len(sequence_rows) if sequence_rows else 0.0
        ),
        failure_count=sum(row.failure is not None for row in rows),
        mean_normalized_edit_diversity=sum(row.normalized_edit_diversity for row in rows) / count,
        mean_kmer_diversity=sum(row.kmer_diversity for row in rows) / count,
        mean_nearest_training_kmer_similarity=(
            sum(nearest_training) / len(nearest_training) if nearest_training else None
        ),
        mean_max_homopolymer_run=sum(row.max_homopolymer_run for row in rows) / count,
        maximum_homopolymer_run=max(row.max_homopolymer_run for row in rows),
        runtime_seconds=sum(row.runtime_seconds for row in rows),
        peak_cuda_memory_bytes=max(peak_memory) if peak_memory else None,
        rnafold_ok_rate=(
            sum(row.rnafold_status == "ok" for row in rnafold_rows) / len(rnafold_rows) if rnafold_rows else None
        ),
        mean_rnafold_mfe_kcal_mol=_mean_available(row.rnafold_mfe_kcal_mol for row in rows),
        mean_rnafold_paired_fraction=_mean_available(row.rnafold_paired_fraction for row in rows),
        mean_rnafold_motif_paired_fraction=_mean_available(row.rnafold_motif_paired_fraction for row in rows),
        rnafold_failure_count=sum(row.rnafold_status not in {None, "ok"} for row in rows),
    )


def _mean_available(values) -> float | None:
    available = [value for value in values if value is not None]
    return sum(available) / len(available) if available else None


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


def maximum_homopolymer_run(sequence: str) -> int:
    if not sequence:
        return 0
    maximum = 1
    current = 1
    for previous, base in pairwise(sequence):
        current = current + 1 if base == previous else 1
        maximum = max(maximum, current)
    return maximum


def within_group_diversity(
    sequences: list[str],
    k: int = 5,
) -> tuple[list[float], list[float]]:
    if k <= 0:
        raise ValueError("k must be positive")
    edit_scores: list[float] = []
    kmer_scores: list[float] = []
    for index, sequence in enumerate(sequences):
        peers = [peer for peer_index, peer in enumerate(sequences) if peer_index != index]
        if not peers:
            edit_scores.append(0.0)
            kmer_scores.append(0.0)
            continue
        edit_scores.append(sum(normalized_edit_distance(sequence, peer) for peer in peers) / len(peers))
        kmer_scores.append(sum(1.0 - kmer_jaccard(sequence, peer, k=k) for peer in peers) / len(peers))
    return edit_scores, kmer_scores


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


def paired_bootstrap_by_motif(
    left_by_motif: dict[str, float],
    right_by_motif: dict[str, float],
    seed: int = 42,
    samples: int = 10000,
) -> tuple[tuple[str, ...], BootstrapDifference]:
    common_motif_ids = tuple(sorted(set(left_by_motif) & set(right_by_motif)))
    if not common_motif_ids:
        raise ValueError("paired methods must share at least one motif ID")
    difference = paired_bootstrap(
        [left_by_motif[motif_id] for motif_id in common_motif_ids],
        [right_by_motif[motif_id] for motif_id in common_motif_ids],
        seed=seed,
        samples=samples,
    )
    return common_motif_ids, difference
