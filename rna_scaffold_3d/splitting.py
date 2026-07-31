from __future__ import annotations

import hashlib
import json
import math
import random
import csv
import copy
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from torch.utils.data import Subset


SPLIT_FORMAT_VERSION = 3
_UINT64_MASK = (1 << 64) - 1


def _normalise_sequence(sequence: str) -> str:
    return str(sequence).upper().replace("T", "U")


def _pdb_group(target_id: str) -> str:
    return target_id.rsplit("_", 1)[0] if "_" in target_id else target_id


def _kmer_counts(sequence: str, kmer_size: int) -> Counter[int]:
    sequence = _normalise_sequence(sequence)
    if len(sequence) < kmer_size:
        tokens = [sequence]
    else:
        tokens = (sequence[index:index + kmer_size] for index in range(len(sequence) - kmer_size + 1))
    return Counter(
        int.from_bytes(hashlib.blake2b(token.encode("ascii"), digest_size=8).digest(), "little")
        for token in tokens
    )


def _minhash_signature(kmers: set[int], num_hashes: int) -> tuple[int, ...]:
    # Deterministic universal-hash permutations avoid Python's process-randomised hash().
    signature: list[int] = []
    for index in range(num_hashes):
        a = (0x9E3779B185EBCA87 + 2 * index) | 1
        b = (0xC2B2AE3D27D4EB4F * (index + 1)) & _UINT64_MASK
        signature.append(min(((a * value + b) & _UINT64_MASK) for value in kmers))
    return tuple(signature)


def _jaccard(left: Counter[int], right: Counter[int]) -> float:
    # Counter intersection is implemented in C and is materially faster than
    # walking the Python key union during exhaustive split construction.
    intersection_size = sum((left & right).values())
    union_size = sum(left.values()) + sum(right.values()) - intersection_size
    return intersection_size / union_size if union_size else 1.0


def _length_compatible_pairs(
    kmers: Sequence[Counter[int]],
    jaccard_threshold: float,
):
    """Yield every pair whose weighted-Jaccard upper bound reaches threshold."""
    totals = [sum(value.values()) for value in kmers]
    ordered = sorted(range(len(kmers)), key=lambda index: (totals[index], index))
    for position, left in enumerate(ordered):
        left_total = totals[left]
        for right in ordered[position + 1:]:
            right_total = totals[right]
            if left_total / right_total < jaccard_threshold:
                break
            yield left, right


def _audit_cross_partition(
    kmers: Sequence[Counter[int]],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
) -> tuple[int, float]:
    """Exactly inspect every train×validation pair."""
    max_similarity = 0.0
    for left in train_indices:
        for right in val_indices:
            max_similarity = max(
                max_similarity, _jaccard(kmers[left], kmers[right])
            )
    return len(train_indices) * len(val_indices), max_similarity


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


@dataclass(frozen=True)
class SplitAudit:
    num_records: int
    num_clusters: int
    train_records: int
    val_records: int
    exact_sequence_overlap: int
    cross_split_pairs_checked: int
    cross_split_audit_exhaustive: bool
    max_cross_split_jaccard: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "num_records": self.num_records,
            "num_clusters": self.num_clusters,
            "train_records": self.train_records,
            "val_records": self.val_records,
            "exact_sequence_overlap": self.exact_sequence_overlap,
            "cross_split_pairs_checked": self.cross_split_pairs_checked,
            "cross_split_audit_exhaustive": self.cross_split_audit_exhaustive,
            "max_cross_split_jaccard": self.max_cross_split_jaccard,
        }


@dataclass(frozen=True)
class HoldoutAudit:
    training_records_before: int
    training_records_after: int
    holdout_records: int
    excluded_records: int
    exact_sequence_exclusions: int
    near_duplicate_exclusions: int
    jaccard_threshold: float
    cross_pairs_checked: int
    cross_pair_audit_exhaustive: bool

    def as_dict(self) -> dict[str, int | float]:
        return {
            "training_records_before": self.training_records_before,
            "training_records_after": self.training_records_after,
            "holdout_records": self.holdout_records,
            "excluded_records": self.excluded_records,
            "exact_sequence_exclusions": self.exact_sequence_exclusions,
            "near_duplicate_exclusions": self.near_duplicate_exclusions,
            "jaccard_threshold": self.jaccard_threshold,
            "cross_pairs_checked": self.cross_pairs_checked,
            "cross_pair_audit_exhaustive": self.cross_pair_audit_exhaustive,
        }


@dataclass(frozen=True)
class _SequenceOnlyRecord:
    target_id: str
    sequence: str


def sequence_dataset_fingerprint(records: Sequence[object]) -> str:
    """Hash ordered target IDs and normalized sequences for split provenance."""
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record.target_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_normalise_sequence(record.sequence).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


# Private compatibility alias for existing internal call sites.
_dataset_fingerprint = sequence_dataset_fingerprint


def _candidate_pairs(signatures: Sequence[tuple[int, ...]], band_size: int) -> set[tuple[int, int]]:
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for index, signature in enumerate(signatures):
        for band_start in range(0, len(signature), band_size):
            key = (band_start // band_size, signature[band_start:band_start + band_size])
            buckets.setdefault(key, []).append(index)
    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        for offset, left in enumerate(members):
            for right in members[offset + 1:]:
                pairs.add((left, right))
    return pairs


def _anchor_candidate_pairs(records: Sequence[object], anchor_size: int = 16) -> set[tuple[int, int]]:
    """Add near-identical candidates that set MinHash can miss on repetitive RNA."""
    buckets: dict[tuple[int, int, str], list[int]] = {}
    for index, record in enumerate(records):
        sequence = _normalise_sequence(record.sequence)
        width = min(anchor_size, len(sequence))
        max_start = max(0, len(sequence) - width)
        starts = (0, max_start // 3, (2 * max_start) // 3, max_start)
        # A 10%-wide length bucket permits small indels but avoids broad prefix buckets.
        length_bucket = round(math.log(max(1, len(sequence)), 1.1))
        for anchor_index, start in enumerate(starts):
            key = (length_bucket, anchor_index, sequence[start:start + width])
            buckets.setdefault(key, []).append(index)
    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        for offset, left in enumerate(members):
            for right in members[offset + 1:]:
                pairs.add((left, right))
    return pairs


def validate_split_manifest_partitions(
    payload: dict[str, object],
    records: Sequence[object],
    *,
    kmer_size: int,
    jaccard_threshold: float,
) -> tuple[list[int], list[int], SplitAudit]:
    """Fail closed when a matching manifest has inconsistent partitions."""
    partitions: dict[str, list[int]] = {}
    for name in ("train", "val"):
        raw_indices = payload.get(f"{name}_indices")
        raw_target_ids = payload.get(f"{name}_target_ids")
        if not isinstance(raw_indices, list) or not isinstance(
            raw_target_ids, list
        ):
            raise ValueError(
                f"Matching split manifest lacks {name} indices/target IDs."
            )
        if len(raw_indices) != len(raw_target_ids):
            raise ValueError(
                f"Matching split manifest {name} indices/target IDs "
                "differ in length."
            )
        if any(type(index) is not int for index in raw_indices):
            raise ValueError(
                f"Matching split manifest {name} indices must be integers."
            )
        indices = [int(index) for index in raw_indices]
        expected_ids = [
            str(records[index].target_id)
            for index in indices
            if 0 <= index < len(records)
        ]
        if len(expected_ids) != len(indices):
            raise ValueError(
                f"Matching split manifest {name} contains out-of-range indices."
            )
        if expected_ids != [str(value) for value in raw_target_ids]:
            raise ValueError(
                f"Matching split manifest {name} target IDs do not match "
                "the indexed dataset records."
            )
        partitions[name] = indices
    train_indices = partitions["train"]
    val_indices = partitions["val"]
    combined = train_indices + val_indices
    if not train_indices or not val_indices:
        raise ValueError(
            "Matching split manifest must have non-empty train and val partitions."
        )
    if (
        len(set(combined)) != len(combined)
        or sorted(combined) != list(range(len(records)))
    ):
        raise ValueError(
            "Matching split manifest indices must be unique and cover 0..N-1."
        )
    train_pdbs = {
        _pdb_group(str(records[index].target_id))
        for index in train_indices
    }
    val_pdbs = {
        _pdb_group(str(records[index].target_id))
        for index in val_indices
    }
    if train_pdbs & val_pdbs:
        raise ValueError(
            "Matching split manifest leaks a PDB group across partitions."
        )
    train_sequences = {
        _normalise_sequence(records[index].sequence)
        for index in train_indices
    }
    val_sequences = {
        _normalise_sequence(records[index].sequence)
        for index in val_indices
    }
    if train_sequences & val_sequences:
        raise ValueError(
            "Matching split manifest leaks an exact sequence across partitions."
        )
    kmers = [
        _kmer_counts(record.sequence, kmer_size) for record in records
    ]
    checked_pairs, max_cross = _audit_cross_partition(
        kmers, train_indices, val_indices
    )
    if max_cross >= jaccard_threshold:
        raise ValueError(
            "Matching split manifest leaks a near-duplicate sequence "
            "across partitions."
        )
    raw_audit = payload.get("audit")
    if not isinstance(raw_audit, dict):
        raise ValueError("Matching split manifest lacks an audit mapping.")
    try:
        audit = SplitAudit(**raw_audit)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Matching split manifest has an invalid audit mapping."
        ) from error
    if (
        audit.num_records != len(records)
        or audit.train_records != len(train_indices)
        or audit.val_records != len(val_indices)
        or audit.exact_sequence_overlap != 0
        or not audit.cross_split_audit_exhaustive
        or audit.cross_split_pairs_checked != checked_pairs
        or not math.isclose(
            audit.max_cross_split_jaccard,
            max_cross,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "Matching split manifest audit counts disagree with partitions."
        )
    return train_indices, val_indices, audit


def leakage_safe_train_val_split(
    dataset,
    val_fraction: float,
    seed: int,
    *,
    manifest_path: str | Path | None = None,
    kmer_size: int = 8,
    jaccard_threshold: float = 0.8,
):
    """Split whole PDB/near-duplicate clusters and persist an auditable manifest.

    This is an alignment-free near-duplicate guard, not a biological homology
    classifier. Every pair capable of reaching the configured weighted k-mer
    Jaccard threshold is checked exactly; there is no probabilistic LSH stage.
    """
    records = getattr(dataset, "records", None)
    if records is None:
        raise ValueError("Leakage-safe splitting requires dataset.records.")
    if not records or len(records) <= 1 or val_fraction <= 0:
        audit = SplitAudit(
            len(records), len(records), len(records), len(records), 0, 0, True, 0.0
        )
        return dataset, dataset, audit
    if not 0.0 < jaccard_threshold <= 1.0:
        raise ValueError("jaccard_threshold must be in (0, 1].")
    if kmer_size <= 0:
        raise ValueError("kmer_size must be positive.")

    fingerprint = _dataset_fingerprint(records)
    manifest = Path(manifest_path) if manifest_path else None
    if manifest and manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        expected = {
            "format_version": SPLIT_FORMAT_VERSION,
            "dataset_fingerprint": fingerprint,
            "seed": seed,
            "val_fraction": val_fraction,
            "kmer_size": kmer_size,
            "jaccard_threshold": jaccard_threshold,
            "candidate_strategy": "exhaustive_length_bounded",
        }
        if all(payload.get(key) == value for key, value in expected.items()):
            train_indices, val_indices, audit = (
                validate_split_manifest_partitions(
                    payload,
                    records,
                    kmer_size=kmer_size,
                    jaccard_threshold=jaccard_threshold,
                )
            )
            return Subset(dataset, train_indices), Subset(dataset, val_indices), audit

    kmers = [_kmer_counts(record.sequence, kmer_size) for record in records]
    union_find = _UnionFind(len(records))

    pdb_representatives: dict[str, int] = {}
    exact_representatives: dict[str, int] = {}
    for index, record in enumerate(records):
        group = _pdb_group(str(record.target_id))
        if group in pdb_representatives:
            union_find.union(index, pdb_representatives[group])
        else:
            pdb_representatives[group] = index
        sequence = _normalise_sequence(record.sequence)
        if sequence in exact_representatives:
            union_find.union(index, exact_representatives[sequence])
        else:
            exact_representatives[sequence] = index
    for left, right in _length_compatible_pairs(kmers, jaccard_threshold):
        if _jaccard(kmers[left], kmers[right]) >= jaccard_threshold:
            union_find.union(left, right)

    clusters: dict[int, list[int]] = {}
    for index in range(len(records)):
        clusters.setdefault(union_find.find(index), []).append(index)
    cluster_members = sorted(clusters.values(), key=lambda members: tuple(members))
    random.Random(seed).shuffle(cluster_members)
    target_val_size = max(1, round(len(records) * val_fraction))
    val_indices: list[int] = []
    for members in cluster_members:
        if len(val_indices) >= target_val_size:
            break
        val_indices.extend(members)
    val_set = set(val_indices)
    train_indices = [index for index in range(len(records)) if index not in val_set]
    if not train_indices:
        # Keep the smallest cluster as validation and all remaining clusters as training.
        cluster_members.sort(key=lambda members: (len(members), members))
        val_indices = list(cluster_members[0])
        val_set = set(val_indices)
        train_indices = [index for index in range(len(records)) if index not in val_set]
    if not train_indices:
        raise ValueError("All records form one leakage cluster; a leakage-safe train/val split is impossible.")

    train_sequences = {_normalise_sequence(records[index].sequence) for index in train_indices}
    val_sequences = {_normalise_sequence(records[index].sequence) for index in val_indices}
    total_cross_pairs, max_cross = _audit_cross_partition(
        kmers, train_indices, val_indices
    )
    audit = SplitAudit(
        num_records=len(records),
        num_clusters=len(clusters),
        train_records=len(train_indices),
        val_records=len(val_indices),
        exact_sequence_overlap=len(train_sequences & val_sequences),
        cross_split_pairs_checked=total_cross_pairs,
        cross_split_audit_exhaustive=True,
        max_cross_split_jaccard=max_cross,
    )
    if manifest:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": SPLIT_FORMAT_VERSION,
            "dataset_fingerprint": fingerprint,
            "seed": seed,
            "val_fraction": val_fraction,
            "kmer_size": kmer_size,
            "jaccard_threshold": jaccard_threshold,
            "candidate_strategy": "exhaustive_length_bounded",
            "train_indices": train_indices,
            "val_indices": val_indices,
            "train_target_ids": [str(records[index].target_id) for index in train_indices],
            "val_target_ids": [str(records[index].target_id) for index in val_indices],
            "audit": audit.as_dict(),
        }
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return Subset(dataset, train_indices), Subset(dataset, val_indices), audit


def exclude_external_holdout(
    dataset,
    holdout_sequences_csv: str | Path,
    *,
    jaccard_threshold: float = 0.8,
    kmer_size: int = 8,
    manifest_path: str | Path | None = None,
):
    """Remove exact/near-duplicate external holdout sequences before splitting."""
    records = getattr(dataset, "records", None)
    if records is None:
        raise ValueError("External holdout exclusion requires dataset.records.")
    with Path(holdout_sequences_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"target_id", "sequence"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "Holdout CSV must contain target_id and sequence columns."
            )
        holdout = [
            _SequenceOnlyRecord(
                str(row["target_id"]), _normalise_sequence(row["sequence"])
            )
            for row in reader
            if str(row.get("sequence", "")).strip()
        ]
    if not holdout:
        raise ValueError("External holdout CSV contains no RNA sequences.")
    training = [
        _SequenceOnlyRecord(str(record.target_id), _normalise_sequence(record.sequence))
        for record in records
    ]
    training_count = len(records)
    excluded: dict[int, dict[str, object]] = {}
    training_kmers = [
        _kmer_counts(record.sequence, kmer_size) for record in training
    ]
    holdout_kmers = [
        _kmer_counts(record.sequence, kmer_size) for record in holdout
    ]
    # This boundary defines whether external validation is genuinely held out.
    # Unlike the much larger internal all-pairs split, train×external-holdout
    # is small enough to audit exhaustively. LSH candidates are inappropriate
    # here because their allowed false negatives would become data leakage.
    for training_index, training_record in enumerate(training):
        for holdout_index, holdout_record in enumerate(holdout):
            similarity = _jaccard(
                training_kmers[training_index],
                holdout_kmers[holdout_index],
            )
            if similarity < jaccard_threshold:
                continue
            previous = excluded.get(training_index)
            if previous is None or similarity > float(previous["similarity"]):
                excluded[training_index] = {
                    "training_target_id": training_record.target_id,
                    "holdout_target_id": holdout_record.target_id,
                    "similarity": similarity,
                    "reason": (
                        "exact_sequence"
                        if training_record.sequence == holdout_record.sequence
                        else "near_duplicate"
                    ),
                }

    kept_records = [
        record for index, record in enumerate(records) if index not in excluded
    ]
    if not kept_records:
        raise ValueError(
            "External holdout exclusion removed every training record."
        )
    filtered = copy.copy(dataset)
    filtered.records = kept_records
    exact_count = sum(
        item["reason"] == "exact_sequence" for item in excluded.values()
    )
    audit = HoldoutAudit(
        training_records_before=training_count,
        training_records_after=len(kept_records),
        holdout_records=len(holdout),
        excluded_records=len(excluded),
        exact_sequence_exclusions=exact_count,
        near_duplicate_exclusions=len(excluded) - exact_count,
        jaccard_threshold=jaccard_threshold,
        cross_pairs_checked=training_count * len(holdout),
        cross_pair_audit_exhaustive=True,
    )
    if hasattr(filtered, "stats"):
        filtered.stats = dict(filtered.stats)
        filtered.stats["holdout_excluded"] = len(excluded)
        filtered.stats["accepted"] = len(kept_records)
    if manifest_path:
        manifest = Path(manifest_path)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 2,
            "training_dataset_fingerprint": _dataset_fingerprint(records),
            "holdout_dataset_fingerprint": _dataset_fingerprint(holdout),
            "parameters": {
                "kmer_size": kmer_size,
                "jaccard_threshold": jaccard_threshold,
                "candidate_strategy": "exhaustive_cross_product",
            },
            "audit": audit.as_dict(),
            "exclusions": [
                excluded[index] for index in sorted(excluded)
            ],
        }
        manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return filtered, audit
