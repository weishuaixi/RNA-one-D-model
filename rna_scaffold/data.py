from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from rna_scaffold.tokenizer import RnaTokenizer
from rna_scaffold.utils import validate_rna_sequence
from rna_scaffold.records import RnaSequenceRecord
from rna_scaffold.splits import SplitManifest, build_family_disjoint_manifest

DEFAULT_MOTIF_LENGTH_BUCKETS = (
    (8, 15, 0.15),
    (16, 31, 0.30),
    (32, 63, 0.30),
    (64, 127, 0.20),
    (128, 256, 0.05),
)

_RNA_RESIDUE_TO_BASE = {
    "A": "A",
    "C": "C",
    "G": "G",
    "U": "U",
    "RA": "A",
    "RC": "C",
    "RG": "G",
    "RU": "U",
}


@dataclass(frozen=True)
class ScaffoldExample:
    motif: str
    left_sequence: str
    right_sequence: str


@dataclass(frozen=True)
class MaskedScaffoldExample:
    """RNA scaffold inpainting example with functional bases kept fixed."""

    masked_sequence: str
    target_sequence: str
    fixed_positions: tuple[int, ...]

    @property
    def fixed_sequence(self) -> str:
        return "".join(self.target_sequence[position] for position in self.fixed_positions)

    @classmethod
    def from_mask_pattern(cls, target_sequence: str, mask_pattern: str) -> "MaskedScaffoldExample":
        target = target_sequence.strip().upper().replace("T", "U")
        pattern = mask_pattern.strip().upper().replace("T", "U")
        if len(target) != len(pattern):
            raise ValueError("target_sequence and mask_pattern must have the same length.")
        if not validate_rna_sequence(target):
            raise ValueError("target_sequence must contain only A, U, C, and G.")

        masked_tokens: list[str] = []
        fixed_positions: list[int] = []
        for index, marker in enumerate(pattern):
            if marker == "X":
                masked_tokens.append("<MASK>")
                continue
            if marker not in "AUCG":
                raise ValueError("mask_pattern must contain only A, U, C, G, or X.")
            if marker != target[index]:
                raise ValueError("Fixed bases in mask_pattern must match target_sequence.")
            masked_tokens.append(marker)
            fixed_positions.append(index)

        if not fixed_positions:
            raise ValueError("mask_pattern must keep at least one functional motif base fixed.")
        return cls(
            masked_sequence="".join(masked_tokens),
            target_sequence=target,
            fixed_positions=tuple(fixed_positions),
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
    min_motif_length: int = 8,
    max_motif_length: int = 256,
    motif_length_buckets: tuple[tuple[int, int, float], ...] | list[list[float]] | None = None,
    min_flank_length: int = 4,
    min_total_scaffold_length: int = 24,
) -> MotifScaffoldExample:
    """Sample a reproducible, non-fixed motif while leaving scaffold context."""
    if min_motif_length < 1:
        raise ValueError("min_motif_length must be positive")
    if min_flank_length < 0:
        raise ValueError("min_flank_length must be non-negative")
    if min_total_scaffold_length < 0:
        raise ValueError("min_total_scaffold_length must be non-negative")
    required_context = max(2 * min_flank_length, min_total_scaffold_length)
    largest = min(max_motif_length, len(record.sequence) - required_context)
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
        min_motif_length: int = 8,
        max_motif_length: int = 256,
        motif_length_buckets: tuple[tuple[int, int, float], ...] | list[list[float]] | None = DEFAULT_MOTIF_LENGTH_BUCKETS,
        min_flank_length: int = 4,
        min_total_scaffold_length: int = 24,
        seed: int = 42,
        allow_empty: bool = False,
    ) -> None:
        required_context = max(2 * min_flank_length, min_total_scaffold_length)
        self.input_record_count = len(records)
        self.records = [
            record
            for record in records
            if min_motif_length + required_context <= len(record.sequence) <= max_length
        ]
        self.excluded_record_count = self.input_record_count - len(self.records)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_motif_length = min_motif_length
        self.max_motif_length = max_motif_length
        self.motif_length_buckets = motif_length_buckets
        self.min_flank_length = min_flank_length
        self.min_total_scaffold_length = min_total_scaffold_length
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
        example = sample_motif_example(
            record,
            generator,
            self.min_motif_length,
            self.max_motif_length,
            self.motif_length_buckets,
            self.min_flank_length,
            self.min_total_scaffold_length,
        )
        length = example.total_length
        target = torch.full((self.max_length,), self.tokenizer.pad_token_id, dtype=torch.long)
        target[:length] = torch.tensor(self.tokenizer.encode(example.target_sequence))
        base_to_class = {"A": 0, "U": 1, "C": 2, "G": 3}
        target_base_ids = torch.full((self.max_length,), -100, dtype=torch.long)
        target_base_ids[:length] = torch.tensor(
            [base_to_class[base] for base in example.target_sequence], dtype=torch.long
        )
        input_ids = torch.full((self.max_length,), self.tokenizer.pad_token_id, dtype=torch.long)
        input_ids[:length] = self.tokenizer.token_to_id[self.tokenizer.special.mask]
        fixed_mask = torch.zeros(self.max_length, dtype=torch.bool)
        fixed_mask[example.motif_start : example.motif_end] = True
        input_ids[fixed_mask] = target[fixed_mask]
        attention_mask = torch.arange(self.max_length) < length
        return {
            "input_ids": input_ids,
            "target_token_ids": target,
            "target_base_ids": target_base_ids,
            "fixed_mask": fixed_mask,
            "attention_mask": attention_mask,
            "target_length": torch.tensor(length, dtype=torch.long),
            "motif_start": torch.tensor(example.motif_start, dtype=torch.long),
        }


def build_partitioned_denoising_datasets(
    records: list[RnaSequenceRecord],
    tokenizer: RnaTokenizer,
    max_length: int = 512,
    min_motif_length: int = 8,
    max_motif_length: int = 256,
    motif_length_buckets: tuple[tuple[int, int, float], ...] | list[list[float]] | None = DEFAULT_MOTIF_LENGTH_BUCKETS,
    min_flank_length: int = 4,
    min_total_scaffold_length: int = 24,
    seed: int = 42,
) -> tuple[dict[str, RnaMotifDenoisingDataset], SplitManifest]:
    manifest = build_family_disjoint_manifest(records, seed=seed)
    by_id = {record.target_id: record for record in records}
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
            seed=seed,
            allow_empty=True,
        )
        for partition in manifest.partitions
    }
    return datasets, manifest


class RnaScaffoldDataset(Dataset):
    """Teacher-forcing dataset for motif-conditioned L/R scaffold generation."""

    def __init__(
        self,
        examples: list[ScaffoldExample],
        tokenizer: RnaTokenizer,
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        source = f"<BOS>{example.motif}<EOS>"
        target = (
            f"<BOS><LEFT>{example.left_sequence}<END_LEFT>"
            f"<RIGHT>{example.right_sequence}<END_RIGHT><EOS>"
        )
        return {
            "input_ids": self._encode_and_pad(source, self.max_source_length),
            "labels": self._encode_and_pad(target, self.max_target_length),
        }

    def _encode_and_pad(self, text: str, max_length: int) -> torch.Tensor:
        ids = self.tokenizer.encode(text)[:max_length]
        ids += [self.tokenizer.pad_token_id] * (max_length - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    @staticmethod
    def examples_from_sequences(
        sequences: list[str],
        motif_length: int,
        stem_length: int,
        min_flank_length: int = 1,
    ) -> list[ScaffoldExample]:
        examples: list[ScaffoldExample] = []
        for raw_sequence in sequences:
            sequence = raw_sequence.strip().upper().replace("T", "U")
            if len(sequence) < motif_length + 2 * min_flank_length:
                continue
            if not validate_rna_sequence(sequence):
                continue
            start = (len(sequence) - motif_length) // 2
            end = start + motif_length
            motif = sequence[start : start + motif_length]
            left = sequence[max(0, start - stem_length) : start]
            right = sequence[end : end + stem_length]
            if len(left) < min_flank_length or len(right) < min_flank_length:
                continue
            examples.append(
                ScaffoldExample(
                    motif=motif,
                    left_sequence=left,
                    right_sequence=right,
                )
            )
        return examples


class RnaMaskedScaffoldDataset(Dataset):
    """Teacher-forcing dataset for motif-fixed RNA scaffold inpainting."""

    def __init__(
        self,
        examples: list[MaskedScaffoldExample],
        tokenizer: RnaTokenizer,
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        source = f"<BOS>{example.masked_sequence}<EOS>"
        target = f"<BOS>{example.target_sequence}<EOS>"
        return {
            "input_ids": self._encode_and_pad(source, self.max_source_length),
            "labels": self._encode_and_pad(target, self.max_target_length),
        }

    def _encode_and_pad(self, text: str, max_length: int) -> torch.Tensor:
        ids = self.tokenizer.encode(text)[:max_length]
        ids += [self.tokenizer.pad_token_id] * (max_length - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    @staticmethod
    def examples_from_sequences(
        sequences: list[str],
        motif_length: int,
        min_flank_length: int = 1,
    ) -> list[MaskedScaffoldExample]:
        examples: list[MaskedScaffoldExample] = []
        for raw_sequence in sequences:
            sequence = raw_sequence.strip().upper().replace("T", "U")
            if len(sequence) < motif_length + 2 * min_flank_length:
                continue
            if not validate_rna_sequence(sequence):
                continue
            start = (len(sequence) - motif_length) // 2
            end = start + motif_length
            mask_pattern = "X" * start + sequence[start:end] + "X" * (len(sequence) - end)
            examples.append(MaskedScaffoldExample.from_mask_pattern(sequence, mask_pattern))
        return examples


def load_fasta_sequences(path: str | Path) -> list[str]:
    sequences: list[str] = []
    current: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                sequences.append("".join(current))
                current = []
            continue
        current.append(line)
    if current:
        sequences.append("".join(current))
    return sequences


def load_csv_sequences(path: str | Path) -> list[str]:
    sequences: list[str] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "sequence" not in reader.fieldnames:
            raise ValueError(f"CSV training data must contain a sequence column: {path}")
        for row in reader:
            sequence = (row.get("sequence") or "").strip()
            if sequence:
                sequences.append(sequence)
    return sequences


def load_pdb_rna_sequences(path: str | Path) -> list[str]:
    """Extract RNA chain sequences from a PDB file.

    Priority:
    1. SEQRES records, because they contain the declared polymer sequence.
    2. ATOM/HETATM residue order, useful for minimal/trimmed PDB files.

    DNA residues and unknown modified bases are ignored by default.
    """
    pdb_path = Path(path)
    text = pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    seqres_chains: dict[str, list[str]] = {}
    for line in text:
        if not line.startswith("SEQRES"):
            continue
        chain_id = line[11].strip() or "_"
        residues = line[19:].split()
        bases = [_RNA_RESIDUE_TO_BASE[residue] for residue in residues if residue in _RNA_RESIDUE_TO_BASE]
        if bases:
            seqres_chains.setdefault(chain_id, []).extend(bases)

    sequences = ["".join(bases) for bases in seqres_chains.values() if bases]
    if sequences:
        return sequences

    atom_chains: dict[str, list[str]] = {}
    seen_residues: set[tuple[str, str, str]] = set()
    for line in text:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        residue = line[17:20].strip()
        base = _RNA_RESIDUE_TO_BASE.get(residue)
        if base is None:
            continue
        chain_id = line[21].strip() or "_"
        residue_number = line[22:27].strip()
        residue_key = (chain_id, residue_number, residue)
        if residue_key in seen_residues:
            continue
        seen_residues.add(residue_key)
        atom_chains.setdefault(chain_id, []).append(base)
    return ["".join(bases) for bases in atom_chains.values() if bases]


def load_sequences(path: str | Path) -> list[str]:
    """Load RNA sequences from FASTA, CSV, a PDB file, or a directory of sequence files."""
    source = Path(path)
    if source.is_dir():
        sequences: list[str] = []
        files = sorted(
            p
            for p in source.rglob("*")
            if p.is_file()
            and (
                p.suffix.lower() in {".pdb", ".ent", ".fa", ".fasta", ".fna"}
                or (p.suffix.lower() == ".csv" and "sequences" in p.name.lower())
            )
        )
        for file_path in files:
            sequences.extend(load_sequences(file_path))
        return sequences

    suffix = source.suffix.lower()
    if suffix in {".pdb", ".ent"}:
        return load_pdb_rna_sequences(source)
    if suffix in {".fa", ".fasta", ".fna"}:
        return load_fasta_sequences(source)
    if suffix == ".csv":
        return load_csv_sequences(source)
    raise ValueError(f"Unsupported training data format: {source}")
