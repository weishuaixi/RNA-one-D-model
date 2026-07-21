from rna_scaffold.generate import (
    RnaTrainingPrior,
    build_auto_masked_scaffold_prompts,
    build_motif_scaffold_sequence,
    build_random_natural_scaffold_result,
    build_single_best_result,
    generate_rna_sequence,
)


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


def test_generate_rna_sequence_returns_only_full_sequence_string_from_motif():
    sequence = generate_rna_sequence(
        motif="GCGG",
        num_candidates=16,
        rng_seed=31,
    )

    assert isinstance(sequence, str)
    assert set(sequence).issubset({"A", "U", "C", "G"})
    assert "GCGG" in sequence
    assert "<MASK>" not in sequence


def test_generate_rna_sequence_can_follow_training_length_distribution_without_user_masks(tmp_path):
    train_csv = tmp_path / "train_sequences.csv"
    train_csv.write_text(
        "target_id,sequence\n"
        "a,AAAAAAAAAAGCGGCCCCCCCCCC\n"
        "b,UUUUUUUUUUGCGGGGGGGGGGGG\n",
        encoding="utf-8",
    )

    sequence = generate_rna_sequence(
        motif="GCGG",
        num_candidates=8,
        rng_seed=4,
        train_data=train_csv,
    )

    assert len(sequence) == 24
    assert "GCGG" in sequence
    assert "<MASK>" not in sequence
