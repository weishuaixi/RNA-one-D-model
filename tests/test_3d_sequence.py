import pytest

from rna_scaffold_3d.sequence import RNA_BASE_TO_ID, encode_rna_sequence, validate_rna_sequence


def test_validate_rna_sequence_accepts_rna_and_normalizes_dna_t():
    assert validate_rna_sequence("AUGC")
    assert validate_rna_sequence("ATGC")


def test_validate_rna_sequence_rejects_empty_or_unknown_bases():
    assert not validate_rna_sequence("")
    assert not validate_rna_sequence("AUGX")


def test_encode_rna_sequence_returns_one_dimensional_training_ids():
    assert encode_rna_sequence("ATGC") == [
        RNA_BASE_TO_ID["A"],
        RNA_BASE_TO_ID["U"],
        RNA_BASE_TO_ID["G"],
        RNA_BASE_TO_ID["C"],
    ]


def test_encode_rna_sequence_rejects_invalid_sequence():
    with pytest.raises(ValueError, match="sequence must contain only A, U, C, and G"):
        encode_rna_sequence("AUGX")
