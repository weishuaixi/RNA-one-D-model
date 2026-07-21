from __future__ import annotations


RNA_BASE_TO_ID = {"A": 1, "U": 2, "C": 3, "G": 4}
RNA_ID_TO_BASE = {index: base for base, index in RNA_BASE_TO_ID.items()}
RNA3D_PAD_ID = 0
RNA3D_MASK_ID = 5


def normalize_rna_sequence(sequence: str) -> str:
    return sequence.strip().upper().replace("T", "U")


def validate_rna_sequence(sequence: str) -> bool:
    normalized = normalize_rna_sequence(sequence)
    return bool(normalized) and set(normalized).issubset({"A", "U", "C", "G"})


def encode_rna_sequence(sequence: str) -> list[int]:
    normalized = normalize_rna_sequence(sequence)
    if not validate_rna_sequence(normalized):
        raise ValueError("sequence must contain only A, U, C, and G.")
    return [RNA_BASE_TO_ID[base] for base in normalized]
