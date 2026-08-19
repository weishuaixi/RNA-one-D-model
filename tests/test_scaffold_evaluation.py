import pytest

from rna_scaffold.evaluation import (
    CandidateMetric,
    normalized_edit_distance,
    paired_bootstrap,
    summarize_candidates,
)


def test_candidate_summary_counts_invalid_and_duplicate_outputs():
    rows = [
        CandidateMetric("m1", "a", True, True, 20, 0.5, None),
        CandidateMetric("m1", "a", True, True, 20, 0.5, None),
        CandidateMetric("m1", "b", False, False, 18, 0.7, "invalid"),
    ]

    summary = summarize_candidates(rows)

    assert summary.valid_rate == pytest.approx(2 / 3)
    assert summary.motif_preservation_rate == pytest.approx(2 / 3)
    assert summary.unique_rate == pytest.approx(2 / 3)
    assert summary.failure_count == 1


def test_normalized_edit_distance_has_expected_bounds():
    assert normalized_edit_distance("AUGC", "AUGC") == 0.0
    assert normalized_edit_distance("AAAA", "UUUU") == 1.0
    assert normalized_edit_distance("", "") == 0.0


def test_paired_bootstrap_is_reproducible_and_paired():
    first = paired_bootstrap([1.0, 2.0, 3.0], [0.0, 1.0, 1.0], seed=42, samples=1000)
    second = paired_bootstrap([1.0, 2.0, 3.0], [0.0, 1.0, 1.0], seed=42, samples=1000)

    assert first == second
    assert first.mean_difference == pytest.approx(4 / 3)
    assert first.lower <= first.mean_difference <= first.upper


def test_paired_bootstrap_rejects_unpaired_inputs():
    with pytest.raises(ValueError, match="equal non-zero length"):
        paired_bootstrap([1.0], [1.0, 2.0])
