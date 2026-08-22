from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from rna_scaffold.geometry import (
    minimum_legal_total_length,
    record_supports_legal_geometry,
    validate_scaffold_length_limit,
)
from rna_scaffold.records import RnaSequenceRecord
from rna_scaffold.splits import SplitManifest, build_cluster_disjoint_manifest
from rna_scaffold.tokenizer import RnaTokenizer

DEFAULT_MOTIF_LENGTH_BUCKETS = (
    (4, 15, 0.15),
    (16, 31, 0.30),
    (32, 63, 0.30),
    (64, 127, 0.20),
    (128, 512, 0.05),
)

@dataclass(frozen=True)
class MotifScaffoldExample:
    motif: str
    target_sequence: str
    motif_start: int

    @property
    def motif_end(self) -> int:
        return self.motif_start + len(self.motif)

    @property
    def total_length(self) -> int:
        return len(self.target_sequence)


def sample_motif_example(
    record: RnaSequenceRecord,
    generator: torch.Generator,
    min_motif_length: int = 4,
    max_motif_length: int | None = None,
    motif_length_buckets: tuple[tuple[int, int, float], ...] | list[list[float]] | None = None,
    min_flank_length: int = 2,
    min_total_scaffold_length: int = 8,
    preferred_total_scaffold_length: int = 24,
) -> MotifScaffoldExample:
    """Sample a reproducible, non-fixed motif while leaving scaffold context."""
    if min_motif_length < 1:
        raise ValueError("min_motif_length must be positive")
    if min_flank_length < 0:
        raise ValueError("min_flank_length must be non-negative")
    if min_total_scaffold_length < 0:
        raise ValueError("min_total_scaffold_length must be non-negative")
    minimum_total = minimum_legal_total_length(
        min_motif_length,
        min_flank_length,
        min_total_scaffold_length,
    )
    if preferred_total_scaffold_length < min_total_scaffold_length:
        raise ValueError(
            "preferred_total_scaffold_length must be at least min_total_scaffold_length"
        )
    if len(record.sequence) < minimum_total:
        raise ValueError("sequence is too short for the requested motif and scaffold context")
    # The preference is a soft total-canvas target. It never becomes extra
    # motif-external context and therefore never excludes a legal shorter row.
    largest = len(record.sequence) - 2 * min_flank_length
    if max_motif_length is not None:
        largest = min(max_motif_length, largest)
    if largest < min_motif_length:
        raise ValueError("sequence is too short for the requested motif and scaffold context")
    motif_length = _sample_motif_length(
        generator,
        min_motif_length,
        largest,
        motif_length_buckets,
    )
    first_start = min_flank_length
    last_start = len(record.sequence) - motif_length - min_flank_length
    motif_start = int(
        torch.randint(first_start, last_start + 1, (1,), generator=generator).item()
    )
    motif = record.sequence[motif_start : motif_start + motif_length]
    return MotifScaffoldExample(motif, record.sequence, motif_start)


def _sample_motif_length(
    generator: torch.Generator,
    minimum: int,
    maximum: int,
    buckets: tuple[tuple[int, int, float], ...] | list[list[float]] | None,
) -> int:
    if not buckets:
        return int(torch.randint(minimum, maximum + 1, (1,), generator=generator).item())
    valid: list[tuple[int, int, float]] = []
    for raw_lower, raw_upper, raw_weight in buckets:
        lower = max(minimum, int(raw_lower))
        upper = min(maximum, int(raw_upper))
        weight = float(raw_weight)
        if weight < 0:
            raise ValueError("motif bucket weights must be non-negative")
        if lower <= upper and weight > 0:
            valid.append((lower, upper, weight))
    if not valid:
        raise ValueError("no motif length bucket overlaps the valid sequence range")
    weights = torch.tensor([bucket[2] for bucket in valid], dtype=torch.float64)
    bucket_index = int(torch.multinomial(weights, 1, generator=generator).item())
    lower, upper, _ = valid[bucket_index]
    return int(torch.randint(lower, upper + 1, (1,), generator=generator).item())


class RnaMotifDenoisingDataset(Dataset):
    """Joint left/right scaffold denoising with immutable motif positions."""

    def __init__(
        self,
        records: list[RnaSequenceRecord],
        tokenizer: RnaTokenizer,
        max_length: int = 512,
        min_motif_length: int = 4,
        max_motif_length: int | None = None,
        motif_length_buckets: tuple[tuple[int, int, float], ...] | list[list[float]] | None = DEFAULT_MOTIF_LENGTH_BUCKETS,
        min_flank_length: int = 2,
        min_total_scaffold_length: int = 8,
        preferred_total_scaffold_length: int = 24,
        full_mask_probability: float = 0.35,
        span_mask_probability: float = 0.35,
        min_random_mask_fraction: float = 0.30,
        max_random_mask_fraction: float = 0.80,
        mean_span_length: int = 8,
        seed: int = 42,
        allow_empty: bool = False,
    ) -> None:
        max_length = validate_scaffold_length_limit(max_length)
        self.input_record_count = len(records)
        self.records = eligible_denoising_records(
            records,
            max_length=max_length,
            min_motif_length=min_motif_length,
            min_flank_length=min_flank_length,
            min_total_scaffold_length=min_total_scaffold_length,
        )
        self.excluded_record_count = self.input_record_count - len(self.records)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_motif_length = min_motif_length
        self.max_motif_length = max_motif_length
        self.motif_length_buckets = motif_length_buckets
        self.min_flank_length = min_flank_length
        self.min_total_scaffold_length = min_total_scaffold_length
        self.preferred_total_scaffold_length = preferred_total_scaffold_length
        if not 0 <= full_mask_probability <= 1:
            raise ValueError("full_mask_probability must be in [0, 1]")
        if not 0 <= span_mask_probability <= 1:
            raise ValueError("span_mask_probability must be in [0, 1]")
        if full_mask_probability + span_mask_probability > 1:
            raise ValueError("full and span mask probabilities must sum to at most one")
        if not 0 < min_random_mask_fraction <= max_random_mask_fraction <= 1:
            raise ValueError("random mask fractions must satisfy 0 < min <= max <= 1")
        if mean_span_length < 2:
            raise ValueError("mean_span_length must be at least two")
        self.full_mask_probability = full_mask_probability
        self.span_mask_probability = span_mask_probability
        self.min_random_mask_fraction = min_random_mask_fraction
        self.max_random_mask_fraction = max_random_mask_fraction
        self.mean_span_length = mean_span_length
        self.seed = seed
        self.epoch = 0
        if not self.records and not allow_empty:
            raise ValueError("No records fit the denoising dataset length limit")

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        generator = torch.Generator().manual_seed(
            self.seed + self.epoch * max(1, len(self.records)) + index
        )
        source_start = 0
        if len(record.sequence) > self.max_length:
            source_start = int(
                torch.randint(
                    0,
                    len(record.sequence) - self.max_length + 1,
                    (1,),
                    generator=generator,
                ).item()
            )
            record = RnaSequenceRecord(
                target_id=record.target_id,
                sequence=record.sequence[source_start : source_start + self.max_length],
                family=record.family,
                source=record.source,
            )
        example = sample_motif_example(
            record,
            generator,
            self.min_motif_length,
            self.max_motif_length,
            self.motif_length_buckets,
            self.min_flank_length,
            self.min_total_scaffold_length,
            self.preferred_total_scaffold_length,
        )
        length = example.total_length
        target = torch.full((self.max_length,), self.tokenizer.pad_token_id, dtype=torch.long)
        target[:length] = torch.tensor(self.tokenizer.encode(example.target_sequence))
        base_to_class = {"A": 0, "U": 1, "C": 2, "G": 3}
        target_base_ids = torch.full((self.max_length,), -100, dtype=torch.long)
        target_base_ids[:length] = torch.tensor(
            [base_to_class[base] for base in example.target_sequence], dtype=torch.long
        )
        input_ids = target.clone()
        fixed_mask = torch.zeros(self.max_length, dtype=torch.bool)
        fixed_mask[example.motif_start : example.motif_end] = True
        attention_mask = torch.arange(self.max_length) < length
        scaffold_mask = attention_mask & ~fixed_mask
        prediction_mask = self._sample_prediction_mask(scaffold_mask, generator)
        input_ids[prediction_mask] = self.tokenizer.token_to_id[self.tokenizer.special.mask]
        return {
            "input_ids": input_ids,
            "target_token_ids": target,
            "target_base_ids": target_base_ids,
            "fixed_mask": fixed_mask,
            "prediction_mask": prediction_mask,
            "attention_mask": attention_mask,
            "target_left_length": torch.tensor(example.motif_start, dtype=torch.long),
            "target_right_length": torch.tensor(
                length - example.motif_end,
                dtype=torch.long,
            ),
            "source_start": torch.tensor(source_start, dtype=torch.long),
        }

    def _sample_prediction_mask(
        self,
        scaffold_mask: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        positions = scaffold_mask.nonzero(as_tuple=False).squeeze(-1)
        if positions.numel() == 0:
            raise ValueError("every example must contain at least one scaffold position")
        draw = float(torch.rand((), generator=generator).item())
        prediction = torch.zeros_like(scaffold_mask)
        if draw < self.full_mask_probability:
            return scaffold_mask.clone()
        if draw < self.full_mask_probability + self.span_mask_probability:
            span_length = min(self.mean_span_length, int(positions.numel()))
            start = int(
                torch.randint(0, int(positions.numel()) - span_length + 1, (1,), generator=generator).item()
            )
            prediction[positions[start : start + span_length]] = True
            return prediction
        fraction = self.min_random_mask_fraction + (
            self.max_random_mask_fraction - self.min_random_mask_fraction
        ) * float(torch.rand((), generator=generator).item())
        count = max(1, min(int(positions.numel()), round(int(positions.numel()) * fraction)))
        order = torch.randperm(int(positions.numel()), generator=generator)[:count]
        prediction[positions[order]] = True
        return prediction


def eligible_denoising_records(
    records: list[RnaSequenceRecord],
    *,
    max_length: int,
    min_motif_length: int,
    min_flank_length: int,
    min_total_scaffold_length: int,
) -> list[RnaSequenceRecord]:
    """Apply the exact V2 geometry eligibility contract before splitting."""
    maximum = validate_scaffold_length_limit(max_length)
    return [
        record
        for record in records
        if record_supports_legal_geometry(
            len(record.sequence),
            max_length=maximum,
            min_motif_length=min_motif_length,
            min_flank_length=min_flank_length,
            min_total_scaffold_length=min_total_scaffold_length,
        )
    ]


def build_partitioned_denoising_datasets(
    records: list[RnaSequenceRecord],
    cluster_by_id: dict[str, str],
    tokenizer: RnaTokenizer,
    max_length: int = 512,
    min_motif_length: int = 4,
    max_motif_length: int | None = None,
    motif_length_buckets: tuple[tuple[int, int, float], ...] | list[list[float]] | None = DEFAULT_MOTIF_LENGTH_BUCKETS,
    min_flank_length: int = 2,
    min_total_scaffold_length: int = 8,
    preferred_total_scaffold_length: int = 24,
    full_mask_probability: float = 0.35,
    span_mask_probability: float = 0.35,
    min_random_mask_fraction: float = 0.30,
    max_random_mask_fraction: float = 0.80,
    mean_span_length: int = 8,
    seed: int = 42,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    source_sha256: dict[str, str] | None = None,
    clustering_thresholds: dict[str, float] | None = None,
    expected_partition_counts: dict[str, int] | None = None,
) -> tuple[dict[str, RnaMotifDenoisingDataset], SplitManifest]:
    record_ids = {record.target_id for record in records}
    if len(record_ids) != len(records):
        raise ValueError("record target IDs must be unique")
    sequence_hashes = [record.sequence_sha256 for record in records]
    if len(sequence_hashes) != len(set(sequence_hashes)):
        raise ValueError("duplicate normalized sequence in formal training records")
    max_length = validate_scaffold_length_limit(max_length)
    eligible_records = eligible_denoising_records(
        records,
        max_length=max_length,
        min_motif_length=min_motif_length,
        min_flank_length=min_flank_length,
        min_total_scaffold_length=min_total_scaffold_length,
    )
    eligible_ids = {record.target_id for record in eligible_records}
    if set(cluster_by_id) != record_ids:
        unknown = sorted(set(cluster_by_id) - record_ids)
        missing = sorted(record_ids - set(cluster_by_id))
        if unknown:
            raise ValueError(f"cluster assignments contain unknown target IDs: {unknown}")
        raise ValueError(f"cluster assignments are incomplete; missing target IDs: {missing}")
    eligible_cluster_by_id = {
        target_id: cluster_id for target_id, cluster_id in cluster_by_id.items() if target_id in eligible_ids
    }
    audit_source_sha256 = dict(source_sha256 or {})
    cluster_manifest_sha256 = getattr(cluster_by_id, "source_sha256", None)
    if cluster_manifest_sha256:
        audit_source_sha256.setdefault("cluster_manifest", cluster_manifest_sha256)
    manifest = build_cluster_disjoint_manifest(
        eligible_records,
        eligible_cluster_by_id,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        source_sha256=audit_source_sha256,
        clustering_thresholds=clustering_thresholds,
        expected_partition_counts=expected_partition_counts,
    )
    by_id = {record.target_id: record for record in eligible_records}
    datasets = {
        partition: RnaMotifDenoisingDataset(
            records=[by_id[target_id] for target_id in manifest.partitions[partition]],
            tokenizer=tokenizer,
            max_length=max_length,
            min_motif_length=min_motif_length,
            max_motif_length=max_motif_length,
            motif_length_buckets=motif_length_buckets,
            min_flank_length=min_flank_length,
            min_total_scaffold_length=min_total_scaffold_length,
            preferred_total_scaffold_length=preferred_total_scaffold_length,
            full_mask_probability=full_mask_probability,
            span_mask_probability=span_mask_probability,
            min_random_mask_fraction=min_random_mask_fraction,
            max_random_mask_fraction=max_random_mask_fraction,
            mean_span_length=mean_span_length,
            seed=seed,
            allow_empty=True,
        )
        for partition in manifest.partitions
    }
    if sum(len(dataset) for dataset in datasets.values()) != manifest.record_counts["total"]:
        raise ValueError("partition totals must equal the eligible dataset total")
    return datasets, manifest
