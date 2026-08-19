from pathlib import Path

import torch

from rna_scaffold.data import (
    MaskedScaffoldExample,
    RnaMotifDenoisingDataset,
    RnaMaskedScaffoldDataset,
    RnaScaffoldDataset,
    ScaffoldExample,
    load_sequences,
)
from rna_scaffold.records import RnaSequenceRecord
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


def test_masked_scaffold_example_keeps_functional_motif_and_masks_scaffold():
    example = MaskedScaffoldExample.from_mask_pattern(
        target_sequence="AAAGCGGUUU",
        mask_pattern="XXXGCGGXXX",
    )

    assert example.target_sequence == "AAAGCGGUUU"
    assert example.masked_sequence == "<MASK><MASK><MASK>GCGG<MASK><MASK><MASK>"
    assert example.fixed_sequence == "GCGG"
    assert example.fixed_positions == (3, 4, 5, 6)


def test_masked_scaffold_dataset_trains_inpainting_from_masked_sequence_to_full_sequence():
    tokenizer = RnaTokenizer()
    example = MaskedScaffoldExample.from_mask_pattern(
        target_sequence="AAAGCGGUUU",
        mask_pattern="XXXGCGGXXX",
    )
    dataset = RnaMaskedScaffoldDataset(
        examples=[example],
        tokenizer=tokenizer,
        max_source_length=32,
        max_target_length=32,
    )

    item = dataset[0]
    source = tokenizer.decode(item["input_ids"].tolist())
    target = tokenizer.decode(item["labels"].tolist())

    assert source.startswith("<BOS><MASK><MASK><MASK>GCGG")
    assert target.startswith("<BOS>AAAGCGGUUU<EOS>")


def test_masked_scaffold_examples_from_sequences_fix_center_motif_by_default():
    examples = RnaMaskedScaffoldDataset.examples_from_sequences(
        sequences=["AAAACCCCUUUU"],
        motif_length=4,
    )

    assert len(examples) == 1
    assert examples[0].target_sequence == "AAAACCCCUUUU"
    assert examples[0].fixed_sequence == "CCCC"
    assert examples[0].masked_sequence == "<MASK><MASK><MASK><MASK>CCCC<MASK><MASK><MASK><MASK>"


def test_denoising_dataset_builds_joint_flank_canvas_and_preserves_motif():
    tokenizer = RnaTokenizer()
    record = RnaSequenceRecord("x", "AAAAGCGGUUUU", "RF1", "unit")
    dataset = RnaMotifDenoisingDataset(
        records=[record],
        tokenizer=tokenizer,
        max_length=32,
        min_motif_length=4,
        max_motif_length=6,
        motif_length_buckets=None,
        min_flank_length=1,
        min_total_scaffold_length=2,
        seed=9,
    )

    item = dataset[0]
    motif_positions = item["fixed_mask"].nonzero().flatten()

    assert item["input_ids"].shape == (32,)
    assert item["attention_mask"].sum().item() == len(record.sequence)
    assert 4 <= motif_positions.numel() <= 6
    assert item["input_ids"][motif_positions].tolist() == (
        item["target_token_ids"][motif_positions].tolist()
    )
    assert item["target_length"].item() == len(record.sequence)
    assert item["motif_start"].item() == motif_positions[0].item()
    assert item["target_base_ids"][:4].tolist() == [0, 0, 0, 0]
    assert item["target_base_ids"][len(record.sequence) :].eq(-100).all()


def test_long_rna_is_retained_and_cropped_reproducibly_per_epoch():
    tokenizer = RnaTokenizer()
    record = RnaSequenceRecord("long", "A" * 300 + "C" * 300, "RF1", "unit")
    first = RnaMotifDenoisingDataset(
        records=[record],
        tokenizer=tokenizer,
        max_length=512,
        min_motif_length=4,
        max_motif_length=None,
        motif_length_buckets=None,
        min_flank_length=2,
        min_total_scaffold_length=8,
        preferred_total_scaffold_length=24,
        seed=17,
    )
    second = RnaMotifDenoisingDataset(
        records=[record],
        tokenizer=tokenizer,
        max_length=512,
        min_motif_length=4,
        max_motif_length=None,
        motif_length_buckets=None,
        min_flank_length=2,
        min_total_scaffold_length=8,
        preferred_total_scaffold_length=24,
        seed=17,
    )

    first_item = first[0]
    second_item = second[0]
    assert len(first) == 1
    assert first_item["target_length"].item() == 512
    assert first_item["source_start"].item() == second_item["source_start"].item()

    first.set_epoch(1)
    assert first[0]["source_start"].item() != first_item["source_start"].item()


def test_denoising_corruption_masks_only_prediction_positions():
    tokenizer = RnaTokenizer()
    record = RnaSequenceRecord("x", "AUGC" * 20, "RF1", "unit")
    dataset = RnaMotifDenoisingDataset(
        [record],
        tokenizer,
        max_length=80,
        min_motif_length=8,
        max_motif_length=8,
        full_mask_probability=0.0,
        span_mask_probability=0.0,
        min_random_mask_fraction=0.5,
        max_random_mask_fraction=0.5,
        seed=9,
    )
    item = dataset[0]
    scaffold = item["attention_mask"] & ~item["fixed_mask"]

    assert item["prediction_mask"].any()
    assert (scaffold & ~item["prediction_mask"]).any()
    assert torch.all(
        item["input_ids"][item["prediction_mask"]]
        == tokenizer.token_to_id[tokenizer.special.mask]
    )
    assert torch.equal(
        item["input_ids"][item["fixed_mask"]],
        item["target_token_ids"][item["fixed_mask"]],
    )


def test_full_mask_corruption_matches_inference_canvas():
    tokenizer = RnaTokenizer()
    record = RnaSequenceRecord("x", "AUGC" * 10, "RF1", "unit")
    dataset = RnaMotifDenoisingDataset(
        [record],
        tokenizer,
        max_length=40,
        full_mask_probability=1.0,
        span_mask_probability=0.0,
        seed=3,
    )
    item = dataset[0]
    scaffold = item["attention_mask"] & ~item["fixed_mask"]

    assert torch.equal(item["prediction_mask"], scaffold)


def test_span_corruption_contains_adjacent_masked_positions():
    tokenizer = RnaTokenizer()
    record = RnaSequenceRecord("x", "AUGC" * 20, "RF1", "unit")
    dataset = RnaMotifDenoisingDataset(
        [record],
        tokenizer,
        max_length=80,
        full_mask_probability=0.0,
        span_mask_probability=1.0,
        mean_span_length=6,
        seed=5,
    )
    prediction = dataset[0]["prediction_mask"]

    assert (prediction[:-1] & prediction[1:]).any()
