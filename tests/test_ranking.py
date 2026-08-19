from rna_scaffold.generate import ScaffoldCandidate
from rna_scaffold.ranking import rank_candidates
from rna_scaffold.validators.rnafold import RnafoldResult


def _candidate(candidate_id, sequence, likelihood, gc, run, entropy):
    return ScaffoldCandidate(
        candidate_id=candidate_id,
        full_sequence=sequence,
        left_sequence=sequence[:2],
        motif="GCGG",
        right_sequence=sequence[6:],
        motif_start=2,
        motif_end=6,
        total_length=len(sequence),
        normalized_log_probability=likelihood,
        checkpoint_sha256="abc",
        seed=42,
        gc_fraction=gc,
        max_homopolymer=run,
        base_entropy=entropy,
        motif_preserved=True,
        valid=True,
        status="ok",
    )


def test_ranking_retains_raw_components_and_penalizes_low_complexity():
    balanced = _candidate("balanced", "AUGCGGAU", -0.3, 0.5, 2, 1.9)
    repetitive = _candidate("repetitive", "GGGCGGGG", -0.2, 0.9, 4, 0.6)
    folds = {
        "balanced": RnafoldResult("ok", "((....))", -2.0, 0.5, 0.0, 0.1, "v", None),
        "repetitive": RnafoldResult("ok", "(((())))", -8.0, 1.0, 1.0, 0.1, "v", None),
    }

    ranked = rank_candidates([repetitive, balanced], folds)

    assert ranked[0].candidate.candidate_id == "balanced"
    assert "mfe_per_nt" in ranked[0].raw_components
    assert "base_entropy" in ranked[0].normalized_components


def test_unavailable_rnafold_is_neutral_and_order_is_deterministic():
    first = _candidate("a", "AUGCGGAU", -0.2, 0.5, 2, 1.8)
    second = _candidate("b", "UUGCGGAA", -0.4, 0.5, 2, 1.8)
    unavailable = RnafoldResult("unavailable", None, None, None, None, 0.0, None, "missing")

    left = rank_candidates([second, first], {"a": unavailable, "b": unavailable})
    right = rank_candidates([second, first], {"a": unavailable, "b": unavailable})

    assert left == right
    assert left[0].candidate.candidate_id == "a"
