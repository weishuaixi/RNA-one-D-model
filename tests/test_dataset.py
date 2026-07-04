from pathlib import Path

from rna_scaffold.data import RnaScaffoldDataset, ScaffoldExample, load_sequences
from rna_scaffold.tokenizer import RnaTokenizer
from rna_scaffold.utils import complementarity_rate, reverse_complement


def test_dataset_builds_variable_length_stem_target_from_full_sequence():
    tokenizer = RnaTokenizer()
    dataset = RnaScaffoldDataset(
        examples=[
            ScaffoldExample(
                motif="AUGCGUACGA",
                left_sequence="AUGCAUGCAU",
                right_sequence=reverse_complement("AUGCAUGCAU"),
            )
        ],
        tokenizer=tokenizer,
        max_source_length=32,
        max_target_length=64,
    )

    item = dataset[0]

    assert item["input_ids"].ndim == 1
    assert item["labels"].ndim == 1
    assert complementarity_rate("AUGCAUGCAU", reverse_complement("AUGCAUGCAU")) >= 0.9


def test_dataset_can_create_examples_from_plain_rna_sequences():
    examples = RnaScaffoldDataset.examples_from_sequences(
        sequences=["AAAACCCCGGGGGUUUUAAAA"],
        motif_length=5,
        stem_length=8,
    )

    assert len(examples) == 1
    assert examples[0].left_sequence == "AAAACCCC"
    assert examples[0].motif == "GGGGG"
    assert examples[0].right_sequence == "UUUUAAAA"


def test_dataset_uses_available_flank_lengths_from_training_sequences():
    examples = RnaScaffoldDataset.examples_from_sequences(
        sequences=["AAACCCGGG"],
        motif_length=3,
        stem_length=8,
        min_flank_length=3,
    )

    assert len(examples) == 1
    assert examples[0].left_sequence == "AAA"
    assert examples[0].motif == "CCC"
    assert examples[0].right_sequence == "GGG"


def test_load_sequences_accepts_kaggle_sequence_csv(tmp_path: Path):
    csv_path = tmp_path / "train_sequences.csv"
    csv_path.write_text(
        "target_id,sequence,temporal_cutoff,description,all_sequences\n"
        "rna_1,AAACCCGGG,2024-01-01,example,\n"
        "rna_2,AAAXXX,2024-01-01,bad,\n",
        encoding="utf-8",
    )

    assert load_sequences(csv_path) == ["AAACCCGGG", "AAAXXX"]
