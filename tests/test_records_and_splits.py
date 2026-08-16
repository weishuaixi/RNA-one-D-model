import pytest
import torch

from rna_scaffold.data import build_partitioned_denoising_datasets, sample_motif_example
from rna_scaffold.records import RnaSequenceRecord, load_sequence_records
from rna_scaffold.splits import build_family_disjoint_manifest, validate_manifest
from rna_scaffold.tokenizer import RnaTokenizer


def test_record_normalizes_thymine_and_whitespace():
    record = RnaSequenceRecord("a", "  atgcau  ", "RF1", "unit")

    assert record.sequence == "AUGCAU"


def test_record_rejects_ambiguous_or_too_long_sequences():
    with pytest.raises(ValueError, match="invalid RNA sequence"):
        RnaSequenceRecord("ambiguous", "AUGN", None, "unit")
    with pytest.raises(ValueError, match="invalid RNA sequence"):
        RnaSequenceRecord("long", "A" * 513, None, "unit")


def test_csv_record_loader_keeps_family_and_source_metadata(tmp_path):
    source = tmp_path / "rfam.csv"
    source.write_text(
        "target_id,sequence,family,source\n"
        "x,ATGCAU,RF00001,Rfam\n",
        encoding="utf-8",
    )

    assert load_sequence_records(source) == [
        RnaSequenceRecord("x", "AUGCAU", "RF00001", "Rfam")
    ]


def test_family_members_never_cross_partitions():
    records = [
        RnaSequenceRecord("a", "AUGCAUGC", "RF1", "unit"),
        RnaSequenceRecord("b", "AUGCAUGG", "RF1", "unit"),
        RnaSequenceRecord("c", "CCCCAAAA", "RF2", "unit"),
    ]

    manifest = build_family_disjoint_manifest(records, seed=7)

    assert manifest.partition_for("a") == manifest.partition_for("b")


def test_manifest_rejects_exact_sequence_overlap():
    records = [
        RnaSequenceRecord("a", "AUGCAUGC", "RF1", "unit"),
        RnaSequenceRecord("b", "AUGCAUGC", "RF2", "unit"),
    ]

    with pytest.raises(ValueError, match="exact sequence overlap"):
        validate_manifest(records, {"train": ["a"], "validation": [], "test": ["b"]})


def test_variable_motif_sampling_is_reproducible_and_not_center_only():
    record = RnaSequenceRecord("a", "AUGCAUGCAUGCAUGCAUGC", "RF1", "unit")
    first_generator = torch.Generator().manual_seed(11)
    second_generator = torch.Generator().manual_seed(11)

    first = [sample_motif_example(record, first_generator, 4, 8) for _ in range(12)]
    second = [sample_motif_example(record, second_generator, 4, 8) for _ in range(12)]

    assert first == second
    assert all(4 <= len(example.motif) <= 8 for example in first)
    assert all(example.target_sequence[example.motif_start : example.motif_end] == example.motif for example in first)
    assert len({example.motif_start for example in first}) > 1
    assert any(example.motif_start != (example.total_length - len(example.motif)) // 2 for example in first)


def test_denoising_datasets_use_manifest_instead_of_random_record_split(tmp_path):
    source = tmp_path / "records.csv"
    source.write_text(
        "target_id,sequence,family,source\n"
        "a,AUGCAUGC,RF1,unit\n"
        "b,AUGCAUGG,RF1,unit\n"
        "c,CCCCAAAA,RF2,unit\n",
        encoding="utf-8",
    )
    records = load_sequence_records(source)
    datasets, manifest = build_partitioned_denoising_datasets(
        records=records,
        tokenizer=RnaTokenizer(),
        min_motif_length=4,
        max_motif_length=6,
        max_length=32,
        seed=7,
    )

    assert manifest.partition_for("a") == manifest.partition_for("b")
    assert sum(len(dataset) for dataset in datasets.values()) == 3
