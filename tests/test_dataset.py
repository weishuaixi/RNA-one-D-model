import torch

from rna_scaffold.data import RnaMotifDenoisingDataset
from rna_scaffold.records import RnaSequenceRecord
from rna_scaffold.tokenizer import RnaTokenizer


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
    assert item["target_base_ids"][:4].tolist() == [0, 0, 0, 0]
    assert item["target_base_ids"][len(record.sequence) :].eq(-100).all()


def test_denoising_dataset_derives_bilateral_targets_without_v1_target_keys():
    dataset = RnaMotifDenoisingDataset(
        records=[RnaSequenceRecord("x", "AACCGGUU", "RF1", "unit")],
        tokenizer=RnaTokenizer(),
        max_length=8,
        min_motif_length=4,
        max_motif_length=4,
        motif_length_buckets=None,
        min_flank_length=2,
        min_total_scaffold_length=0,
        preferred_total_scaffold_length=0,
        seed=9,
    )

    item = dataset[0]

    assert item["target_left_length"].item() == 2
    assert item["target_right_length"].item() == 2
    assert "target_length" not in item
    assert "motif_start" not in item


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
    assert first_item["attention_mask"].sum().item() == 512
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
