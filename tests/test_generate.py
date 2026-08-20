import pytest
import torch

from rna_scaffold.decoding import DecodedScaffold
from rna_scaffold.generate import (
    GenerationSettings,
    RnaTrainingPrior,
    build_auto_masked_scaffold_prompts,
    build_motif_scaffold_sequence,
    build_random_natural_scaffold_result,
    build_single_best_result,
    generate_candidates,
    generate_markov_baseline,
    generate_rna_sequence,
)
from rna_scaffold.model import ScaffoldModelOutput
from rna_scaffold.tokenizer import RnaTokenizer


def test_training_prior_smoothing_keeps_all_bases_sampleable():
    prior = RnaTrainingPrior.from_sequences(["AAAAAAAAAAAA"])

    assert all(probability > 0 for probability in prior.initial.values())
    assert all(
        probability > 0
        for transitions in prior.transition.values()
        for probability in transitions.values()
    )


def test_build_single_best_result_preserves_motif_and_samples_independent_right_flank():
    result = build_single_best_result(
        motif="AUGCGUACGA",
        left_sequence="AUGCAUGCAU",
        quality_score=0.91,
        rng_seed=7,
    )

    assert result.motif == "AUGCGUACGA"
    assert result.full_sequence == result.left_sequence + result.motif + result.right_sequence
    assert result.motif_preserved
    assert len(result.right_sequence) == len(result.left_sequence)
    assert set(result.right_sequence) <= {"A", "U", "C", "G"}
    assert result.left_right_complementarity < 0.8


def test_build_single_best_result_does_not_use_mutation_rate_as_complementarity_target():
    low_mutation = build_single_best_result(
        motif="AUGCGUACGA",
        left_sequence="AUGCAUGCAUAUGCAUGCAU",
        quality_score=0.91,
        mutation_rate=0.0,
        rng_seed=7,
    )
    high_mutation = build_single_best_result(
        motif="AUGCGUACGA",
        left_sequence="AUGCAUGCAUAUGCAUGCAU",
        quality_score=0.91,
        mutation_rate=0.25,
        rng_seed=7,
    )

    assert low_mutation.full_sequence == low_mutation.left_sequence + low_mutation.motif + low_mutation.right_sequence
    assert low_mutation.motif_preserved
    assert low_mutation.right_sequence == high_mutation.right_sequence
    assert low_mutation.left_right_complementarity < 0.8


def test_build_single_best_result_is_reproducible_with_seed():
    first = build_single_best_result(
        motif="AUGCGUACGA",
        left_sequence="AUGCAUGCAUAUGCAUGCAU",
        quality_score=0.91,
        mutation_rate=0.15,
        rng_seed=11,
    )
    second = build_single_best_result(
        motif="AUGCGUACGA",
        left_sequence="AUGCAUGCAUAUGCAUGCAU",
        quality_score=0.91,
        mutation_rate=0.15,
        rng_seed=11,
    )

    assert first.right_sequence == second.right_sequence


def test_build_random_natural_scaffold_result_samples_length_and_preserves_motif():
    result = build_random_natural_scaffold_result(
        motif="AUGCGUACGA",
        min_left_length=12,
        max_left_length=20,
        num_candidates=32,
        rng_seed=3,
    )

    assert result.full_sequence == result.left_sequence + result.motif + result.right_sequence
    assert result.motif_preserved
    assert 12 <= result.left_length <= 20
    assert result.left_length == result.right_length
    assert result.right_sequence != result.left_sequence[::-1]


def test_build_random_natural_scaffold_result_is_reproducible_with_seed():
    first = build_random_natural_scaffold_result(
        motif="AUGCGUACGA",
        min_left_length=12,
        max_left_length=20,
        num_candidates=32,
        rng_seed=5,
    )
    second = build_random_natural_scaffold_result(
        motif="AUGCGUACGA",
        min_left_length=12,
        max_left_length=20,
        num_candidates=32,
        rng_seed=5,
    )

    assert first.full_sequence == second.full_sequence


def test_build_auto_masked_scaffold_prompts_require_only_motif():
    prompts = build_auto_masked_scaffold_prompts(
        motif="GCGG",
        num_candidates=4,
        rng_seed=1,
    )

    assert len(prompts) == 4
    assert all(prompt.motif == "GCGG" for prompt in prompts)
    assert all(prompt.total_length > len(prompt.motif) for prompt in prompts)
    assert all(prompt.masked_sequence.count("<MASK>") == prompt.total_length - len(prompt.motif) for prompt in prompts)
    assert all(prompt.masked_sequence.replace("<MASK>", "") == "GCGG" for prompt in prompts)


def test_build_auto_masked_scaffold_prompts_are_reproducible_and_length_diverse():
    first = build_auto_masked_scaffold_prompts(
        motif="AUGCGUACGA",
        num_candidates=8,
        rng_seed=7,
    )
    second = build_auto_masked_scaffold_prompts(
        motif="AUGCGUACGA",
        num_candidates=8,
        rng_seed=7,
    )

    assert first == second
    assert len({prompt.total_length for prompt in first}) > 1


def test_build_motif_scaffold_sequence_returns_full_rna_from_only_motif():
    result = build_motif_scaffold_sequence(
        motif="GCGG",
        num_candidates=16,
        rng_seed=13,
    )

    assert result.motif == "GCGG"
    assert result.motif in result.full_sequence
    assert result.full_sequence == result.left_sequence + result.motif + result.right_sequence
    assert result.motif_preserved
    assert result.left_length > 0
    assert result.right_length > 0


def test_build_motif_scaffold_sequence_does_not_force_complementary_flanks():
    result = build_motif_scaffold_sequence(
        motif="GCGG",
        num_candidates=16,
        min_total_length=40,
        max_total_length=40,
        rng_seed=13,
    )

    assert result.left_right_complementarity < 0.8


def test_build_motif_scaffold_sequence_can_use_internal_length_range_without_user_masks():
    result = build_motif_scaffold_sequence(
        motif="AUGCGU",
        num_candidates=8,
        min_total_length=30,
        max_total_length=36,
        rng_seed=21,
    )

    assert 30 <= len(result.full_sequence) <= 36
    assert result.full_sequence.count("AUGCGU") == 1


def test_generate_rna_sequence_requires_checkpoint():
    with pytest.raises(TypeError):
        generate_rna_sequence(motif="GCGG")


def test_generate_candidates_derives_canvas_geometry_from_selected_flank_lengths(monkeypatch):
    class PlacementModel(torch.nn.Module):
        max_length = 16

        def __init__(self):
            super().__init__()
            self.model = torch.nn.Identity()

        def forward(self, input_ids, attention_mask=None):
            token_logits = torch.zeros((1, input_ids.shape[1], 4))
            left_length_logits = torch.full((1, 17), -100.0)
            right_length_logits = torch.full((1, 17), -100.0)
            left_length_logits[0, 2] = 100.0
            right_length_logits[0, 3] = 100.0
            return ScaffoldModelOutput(token_logits, left_length_logits, right_length_logits)

    decoded_geometry = []

    def fake_iterative_denoise(model, tokenizer, motif, total_length, motif_start, settings, generator, device):
        decoded_geometry.append((total_length, motif_start))
        return DecodedScaffold(
            sequence="AA" + motif + "AAA",
            normalized_log_probability=-0.5,
            unresolved_counts=(0,),
        )

    loaded = type(
        "Loaded",
        (),
        {
            "model": PlacementModel(),
            "tokenizer": RnaTokenizer(),
            "max_length": 16,
            "checkpoint_sha256": "test-checkpoint",
        },
    )()
    monkeypatch.setattr("rna_scaffold.checkpoints.load_scaffold_checkpoint", lambda checkpoint, device: loaded)
    monkeypatch.setattr("rna_scaffold.decoding.iterative_denoise", fake_iterative_denoise)

    candidates = generate_candidates(
        checkpoint="ignored.ckpt",
        motif="GCGG",
        settings=GenerationSettings(
            num_candidates=1,
            max_attempt_multiplier=1,
            max_length=16,
            min_scaffold_length=1,
            seed=7,
        ),
    )

    assert decoded_geometry == [(9, 2)]
    assert candidates[0].motif_start == 2
    assert candidates[0].total_length == 9


def test_markov_baseline_can_follow_training_length_distribution(tmp_path):
    train_csv = tmp_path / "train_sequences.csv"
    train_csv.write_text(
        "target_id,sequence\n"
        "a,AAAAAAAAAAGCGGCCCCCCCCCC\n"
        "b,UUUUUUUUUUGCGGGGGGGGGGGG\n",
        encoding="utf-8",
    )

    result = generate_markov_baseline(
        motif="GCGG",
        train_data=train_csv,
        seed=4,
    )

    assert len(result.full_sequence) == 24
    assert "GCGG" in result.full_sequence
    assert "<MASK>" not in result.full_sequence


def test_generation_settings_reject_invalid_sampling_values():
    with pytest.raises(ValueError, match="num_candidates"):
        GenerationSettings(num_candidates=0)
