from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from rna_scaffold.tokenizer import RnaTokenizer
from rna_scaffold.utils import validate_rna_sequence

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
