from __future__ import annotations

import hashlib
from pathlib import Path

from torch.utils.data import DataLoader

try:
    import lightning.pytorch as L
except ImportError:  # pragma: no cover
    import pytorch_lightning as L

from rna_scaffold.data import build_partitioned_denoising_datasets
from rna_scaffold.geometry import validate_scaffold_length_limit
from rna_scaffold.records import load_sequence_records
from rna_scaffold.splits import load_cluster_assignments
from rna_scaffold.tokenizer import RnaTokenizer


class RnaScaffoldDataModule(L.LightningDataModule):
    def __init__(
        self,
        tokenizer: RnaTokenizer,
        train_data: str | None = None,
        cluster_manifest: str | None = None,
        task: str = "motif_denoising",
        min_motif_length: int = 4,
        max_motif_length: int | None = None,
        motif_length_buckets: list[list[float]] | None = None,
        min_flank_length: int = 2,
        min_total_scaffold_length: int = 8,
        preferred_total_scaffold_length: int = 24,
        full_mask_probability: float = 0.35,
        span_mask_probability: float = 0.35,
        min_random_mask_fraction: float = 0.30,
        max_random_mask_fraction: float = 0.80,
        mean_span_length: int = 8,
        max_target_length: int = 256,
        batch_size: int = 64,
        num_workers: int = 4,
        val_fraction: float = 0.05,
        test_fraction: float = 0.1,
        expected_partition_counts: dict[str, int] | None = None,
        expected_cluster_count: int | None = None,
        cluster_thresholds: dict[str, float] | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if task != "motif_denoising":
            raise ValueError("task must be 'motif_denoising' in Scaffold Generator V2")
        self.train_data = Path(train_data or "")
        if not str(self.train_data):
            raise ValueError("train_data must be provided")
        self.cluster_manifest = Path(cluster_manifest) if cluster_manifest else None
        if self.cluster_manifest is None:
            raise ValueError("cluster_manifest must be provided for motif_denoising training")
        self.tokenizer = tokenizer
        self.task = task
        self.min_motif_length = min_motif_length
        self.max_motif_length = max_motif_length
        self.motif_length_buckets = motif_length_buckets
        self.min_flank_length = min_flank_length
        self.min_total_scaffold_length = min_total_scaffold_length
        self.preferred_total_scaffold_length = preferred_total_scaffold_length
        self.full_mask_probability = full_mask_probability
        self.span_mask_probability = span_mask_probability
        self.min_random_mask_fraction = min_random_mask_fraction
        self.max_random_mask_fraction = max_random_mask_fraction
        self.mean_span_length = mean_span_length
        self.max_target_length = validate_scaffold_length_limit(
            max_target_length,
            field="max_target_length",
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.expected_partition_counts = expected_partition_counts
        self.expected_cluster_count = expected_cluster_count
        self.cluster_thresholds = cluster_thresholds
        self.seed = seed

    def setup(self, stage: str | None = None) -> None:
        if self.cluster_manifest is None:
            raise ValueError("cluster_manifest must be provided for motif_denoising training")
        cluster_by_id = load_cluster_assignments(self.cluster_manifest)
        records = load_sequence_records(self.train_data, skip_invalid=False)
        datasets, self.split_manifest = build_partitioned_denoising_datasets(
                records=records,
                cluster_by_id=cluster_by_id,
                tokenizer=self.tokenizer,
                max_length=self.max_target_length,
                min_motif_length=self.min_motif_length,
                max_motif_length=self.max_motif_length,
                motif_length_buckets=self.motif_length_buckets,
                min_flank_length=self.min_flank_length,
                min_total_scaffold_length=self.min_total_scaffold_length,
                preferred_total_scaffold_length=self.preferred_total_scaffold_length,
                full_mask_probability=self.full_mask_probability,
                span_mask_probability=self.span_mask_probability,
                min_random_mask_fraction=self.min_random_mask_fraction,
                max_random_mask_fraction=self.max_random_mask_fraction,
                mean_span_length=self.mean_span_length,
                seed=self.seed,
                val_fraction=self.val_fraction,
                test_fraction=self.test_fraction,
                source_sha256={"sequence_table": hashlib.sha256(self.train_data.read_bytes()).hexdigest()},
                clustering_thresholds=self.cluster_thresholds,
                expected_partition_counts=self.expected_partition_counts,
        )
        actual_cluster_count = self.split_manifest.cluster_counts.get("total")
        if (
            self.expected_cluster_count is not None
            and actual_cluster_count != self.expected_cluster_count
        ):
            raise ValueError(
                "eligible cluster count does not match expected: "
                f"actual={actual_cluster_count}, expected={self.expected_cluster_count}"
            )
        self.train_dataset = datasets["train"]
        self.val_dataset = datasets["validation"]
        self.test_dataset = datasets["test"]
        eligible_records = sum(len(dataset) for dataset in datasets.values())
        self.denoising_data_audit = {
                "input_records": len(records),
                "eligible_records": eligible_records,
                "excluded_by_length_or_context": len(records) - eligible_records,
                "min_motif_length": self.min_motif_length,
                "max_motif_length": self.max_motif_length,
                "min_flank_length": self.min_flank_length,
                "min_total_scaffold_length": self.min_total_scaffold_length,
                "preferred_total_scaffold_length": self.preferred_total_scaffold_length,
        }
        print(
            "denoising_data_audit="
            + ", ".join(f"{key}={value}" for key, value in self.denoising_data_audit.items())
        )
        if not len(self.train_dataset):
            raise ValueError("Cluster-disjoint split produced an empty training partition")
        expected_counts = self.expected_partition_counts or {}
        empty_required = [
            partition
            for partition, expected_count in expected_counts.items()
            if expected_count > 0 and not len(datasets[partition])
        ]
        if empty_required:
            raise ValueError(f"expected non-empty partitions are empty: {empty_required}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        dataset = getattr(self, "test_dataset", self.val_dataset)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
