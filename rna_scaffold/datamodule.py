from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

try:
    import lightning.pytorch as L
except ImportError:  # pragma: no cover
    import pytorch_lightning as L

from rna_scaffold.data import RnaScaffoldDataset, load_sequences
from rna_scaffold.tokenizer import RnaTokenizer


class RnaScaffoldDataModule(L.LightningDataModule):
    def __init__(
        self,
        tokenizer: RnaTokenizer,
        train_data: str | None = None,
        train_fasta: str | None = None,
        motif_length: int = 32,
        stem_length: int = 64,
        min_flank_length: int = 1,
        max_source_length: int = 128,
        max_target_length: int = 256,
        batch_size: int = 64,
        num_workers: int = 4,
        val_fraction: float = 0.05,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.train_data = Path(train_data or train_fasta or "")
        if not str(self.train_data):
            raise ValueError("Either train_data or train_fasta must be provided.")
        self.tokenizer = tokenizer
        self.motif_length = motif_length
        self.stem_length = stem_length
        self.min_flank_length = min_flank_length
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.seed = seed

    def setup(self, stage: str | None = None) -> None:
        sequences = load_sequences(self.train_data)
        examples = RnaScaffoldDataset.examples_from_sequences(
            sequences=sequences,
            motif_length=self.motif_length,
            stem_length=self.stem_length,
            min_flank_length=self.min_flank_length,
        )
        if not examples:
            raise ValueError("No valid training examples were built from the training data.")

        dataset = RnaScaffoldDataset(
            examples=examples,
            tokenizer=self.tokenizer,
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
        )
        val_size = max(1, int(len(dataset) * self.val_fraction)) if len(dataset) > 1 else 0
        train_size = len(dataset) - val_size
        if val_size:
            self.train_dataset, self.val_dataset = random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )
        else:
            self.train_dataset = dataset
            self.val_dataset = dataset

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
