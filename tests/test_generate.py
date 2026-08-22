import pytest
import torch

from rna_scaffold.decoding import DecodedScaffold
from rna_scaffold.generate import (
    CandidateDiversityDecision,
    CandidateGenerationAudit,
    GeneratedCandidates,
    GenerationSettings,
    assess_candidate_diversity,
    generate_candidates,
    generate_rna_sequence,
)
from rna_scaffold.model import ScaffoldModelOutput
from rna_scaffold.tokenizer import RnaTokenizer


def test_generate_rna_sequence_requires_checkpoint():
    with pytest.raises(TypeError):
        generate_rna_sequence(motif="GCGG")


def test_generate_candidates_derives_canvas_geometry_from_selected_flank_lengths(monkeypatch):
    class PlacementModel(torch.nn.Module):
        max_length = 10

        def __init__(self):
            super().__init__()
            self.model = torch.nn.Identity()

        def forward(self, input_ids, attention_mask=None):
            token_logits = torch.zeros((1, input_ids.shape[1], 4))
            left_length_logits = torch.full((1, 17), -100.0)
            right_length_logits = torch.full((1, 17), -100.0)
            left_length_logits[0, 2] = 100.0
            right_length_logits[0, 6] = 100.0
            right_length_logits[0, 4] = 90.0
            return ScaffoldModelOutput(token_logits, left_length_logits, right_length_logits)

    decoded_geometry = []

    def fake_iterative_denoise(model, tokenizer, motif, total_length, motif_start, settings, generator, device):
        decoded_geometry.append((total_length, motif_start))
        return DecodedScaffold(
            sequence="AA" + motif + "AAAA",
            normalized_log_probability=-0.5,
            masked_counts=(0,),
        )

    loaded = type(
        "Loaded",
        (),
        {
            "model": PlacementModel(),
            "tokenizer": RnaTokenizer(),
            "max_length": 10,
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

    assert decoded_geometry == [(10, 2)]
    assert candidates[0].motif_start == 2
    assert candidates[0].total_length == 10
    assert len(candidates[0].full_sequence) == candidates[0].total_length


def test_generation_settings_reject_invalid_sampling_values():
    with pytest.raises(ValueError, match="num_candidates"):
        GenerationSettings(num_candidates=0)


def test_candidate_diversity_policy_reports_each_rejection_reason_and_acceptance():
    accepted = ["AUGCGGAU"]

    duplicate = assess_candidate_diversity("AUGCGGAU", accepted, GenerationSettings(num_candidates=1))
    near_duplicate = assess_candidate_diversity(
        "AUGCGGAC",
        accepted,
        GenerationSettings(num_candidates=1, min_normalized_edit_distance=0.25),
    )
    kmer_similar = assess_candidate_diversity(
        "AUGCAGAU",
        accepted,
        GenerationSettings(num_candidates=1, max_kmer_similarity=0.2, kmer_size=2),
    )
    homopolymer = assess_candidate_diversity(
        "AAAAUGCG",
        [],
        GenerationSettings(num_candidates=1, max_homopolymer_run=3),
    )
    acceptable = assess_candidate_diversity("AUGCUGCA", accepted, GenerationSettings(num_candidates=1))

    assert duplicate == CandidateDiversityDecision(False, "duplicate", {})
    assert near_duplicate.reason == "near_duplicate"
    assert kmer_similar.reason == "kmer_similarity"
    assert homopolymer.reason == "homopolymer"
    assert acceptable == CandidateDiversityDecision(True, "accepted", {})


def test_gc_bounds_are_soft_metadata_unless_explicitly_enforced():
    soft = assess_candidate_diversity(
        "GGGGGGGG",
        [],
        GenerationSettings(num_candidates=1, gc_min=0.25, gc_max=0.75),
    )
    hard = assess_candidate_diversity(
        "GGGGGGGG",
        [],
        GenerationSettings(num_candidates=1, gc_min=0.25, gc_max=0.75, enforce_gc_bounds=True),
    )

    assert soft.accepted
    assert soft.reason == "accepted"
    assert soft.metadata == {"gc_in_soft_range": False, "gc_soft_penalty": 0.25}
    assert hard == CandidateDiversityDecision(False, "gc_out_of_bounds", {})


def test_generation_shortfall_is_audited_after_bounded_duplicate_attempts(monkeypatch):
    class PlacementModel(torch.nn.Module):
        max_length = 8

        def __init__(self):
            super().__init__()
            self.model = torch.nn.Identity()

        def forward(self, input_ids, attention_mask=None):
            token_logits = torch.zeros((1, input_ids.shape[1], 4))
            left_length_logits = torch.full((1, 9), -100.0)
            right_length_logits = torch.full((1, 9), -100.0)
            left_length_logits[0, 2] = 100.0
            right_length_logits[0, 2] = 100.0
            return ScaffoldModelOutput(token_logits, left_length_logits, right_length_logits)

    loaded = type(
        "Loaded",
        (),
        {
            "model": PlacementModel(),
            "tokenizer": RnaTokenizer(),
            "max_length": 8,
            "checkpoint_sha256": "test-checkpoint",
        },
    )()
    monkeypatch.setattr("rna_scaffold.checkpoints.load_scaffold_checkpoint", lambda checkpoint, device: loaded)
    monkeypatch.setattr(
        "rna_scaffold.decoding.iterative_denoise",
        lambda *args, **kwargs: DecodedScaffold("AAGCGGUU", -0.5, (0,)),
    )

    result = generate_candidates(
        checkpoint="ignored.ckpt",
        motif="GCGG",
        settings=GenerationSettings(num_candidates=2, max_attempts=3, min_scaffold_length=1, seed=7),
    )

    assert isinstance(result, GeneratedCandidates)
    assert len(result) == 1
    assert result[0].full_sequence == "AAGCGGUU"
    assert result[0].status == "shortfall"
    assert result.audit == CandidateGenerationAudit(
        requested=2,
        max_attempts=3,
        attempted=3,
        accepted=1,
        rejected={"duplicate": 2},
    )


def test_soft_gc_outlier_is_persisted_and_ranked_after_in_range_candidate(monkeypatch):
    class PlacementModel(torch.nn.Module):
        max_length = 8

        def __init__(self):
            super().__init__()
            self.model = torch.nn.Identity()

        def forward(self, input_ids, attention_mask=None):
            token_logits = torch.zeros((1, input_ids.shape[1], 4))
            left_length_logits = torch.full((1, 9), -100.0)
            right_length_logits = torch.full((1, 9), -100.0)
            left_length_logits[0, 2] = 100.0
            right_length_logits[0, 2] = 100.0
            return ScaffoldModelOutput(token_logits, left_length_logits, right_length_logits)

    loaded = type(
        "Loaded",
        (),
        {
            "model": PlacementModel(),
            "tokenizer": RnaTokenizer(),
            "max_length": 8,
            "checkpoint_sha256": "test-checkpoint",
        },
    )()
    decoded = iter((
        DecodedScaffold("AAGCGGUU", -0.9, (0,)),
        DecodedScaffold("GGGCGGGG", -0.1, (0,)),
    ))
    monkeypatch.setattr("rna_scaffold.checkpoints.load_scaffold_checkpoint", lambda checkpoint, device: loaded)
    monkeypatch.setattr("rna_scaffold.decoding.select_flank_lengths", lambda *args, **kwargs: (2, 2))
    monkeypatch.setattr("rna_scaffold.decoding.iterative_denoise", lambda *args, **kwargs: next(decoded))

    result = generate_candidates(
        checkpoint="ignored.ckpt",
        motif="GCGG",
        settings=GenerationSettings(
            num_candidates=2,
            max_attempts=2,
            min_scaffold_length=1,
            gc_min=0.4,
            gc_max=0.6,
        ),
    )

    assert [candidate.full_sequence for candidate in result] == ["AAGCGGUU", "GGGCGGGG"]
    assert result[0].gc_in_soft_range is True
    assert result[0].gc_soft_penalty == 0.0
    assert result[1].gc_in_soft_range is False
    assert result[1].gc_soft_penalty == pytest.approx(0.4)


def test_generated_candidates_remain_a_mutable_list_with_run_metadata():
    audit = CandidateGenerationAudit(requested=1, max_attempts=1, attempted=1, accepted=1, rejected={})
    settings = {"max_attempts": 1}
    generated = GeneratedCandidates([], audit, settings)

    assert isinstance(generated, list)
    assert generated == []
    generated.append("candidate")
    assert generated + ["next"] == ["candidate", "next"]
    assert generated.audit is audit
    assert generated.generation_settings == settings
