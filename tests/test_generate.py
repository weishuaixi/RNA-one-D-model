from rna_scaffold.generate import build_random_natural_scaffold_result, build_single_best_result


def test_build_single_best_result_preserves_motif_and_guarantees_complementarity():
    result = build_single_best_result(
        motif="AUGCGUACGA",
        left_sequence="AUGCAUGCAU",
        quality_score=0.91,
    )

    assert result.motif == "AUGCGUACGA"
    assert result.full_sequence == result.left_sequence + result.motif + result.right_sequence
    assert result.motif_preserved
    assert result.left_right_complementarity >= 0.9


def test_build_single_best_result_can_create_natural_partial_complementarity():
    result = build_single_best_result(
        motif="AUGCGUACGA",
        left_sequence="AUGCAUGCAUAUGCAUGCAU",
        quality_score=0.91,
        mutation_rate=0.15,
        rng_seed=7,
    )

    assert result.full_sequence == result.left_sequence + result.motif + result.right_sequence
    assert result.motif_preserved
    assert 0.8 <= result.left_right_complementarity <= 0.95
    assert result.left_right_complementarity < 1.0


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
    assert 0.75 <= result.left_right_complementarity <= 0.95


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
