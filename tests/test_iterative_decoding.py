import math

import pytest
import torch

from rna_scaffold.decoding import DecodingSettings, iterative_denoise, select_flank_lengths
from rna_scaffold.model import ScaffoldModelOutput
from rna_scaffold.tokenizer import RnaTokenizer


class DeterministicScaffoldModel(torch.nn.Module):
    max_length = 16

    def forward(self, input_ids, attention_mask=None):
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 4), -10.0, device=input_ids.device)
        preferred = torch.arange(length, device=input_ids.device) % 4
        logits.scatter_(2, preferred.view(1, length, 1).expand(batch, -1, -1), 10.0)
        left_length_logits = torch.full((batch, 17), -100.0, device=input_ids.device)
        left_length_logits[:, 3] = 10.0
        right_length_logits = torch.full((batch, 17), -100.0, device=input_ids.device)
        right_length_logits[:, 3] = 10.0
        return ScaffoldModelOutput(logits, left_length_logits, right_length_logits)


class ContextCorrectingScaffoldModel(torch.nn.Module):
    """Makes a low-confidence first guess that context corrects after remasking."""

    max_length = 16

    def __init__(self, mask_id: int, a_id: int) -> None:
        super().__init__()
        self.mask_id = mask_id
        self.a_id = a_id
        self.seen_inputs: list[torch.Tensor] = []

    def forward(self, input_ids, attention_mask=None):
        self.seen_inputs.append(input_ids.detach().cpu().clone())
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 4), -20.0, device=input_ids.device)
        if not bool((input_ids == self.mask_id).any()):
            # The final scoring pass must use these probabilities, rather than
            # probabilities captured while a position was first sampled.
            logits.zero_()
        else:
            preferred = torch.tensor([0, 1, 0, 0, 2, 3], device=input_ids.device)
            if input_ids[0, 0].item() == self.a_id:
                preferred[1] = 2
            logits.scatter_(2, preferred.view(1, length, 1).expand(batch, -1, -1), 20.0)
            if input_ids[0, 0].item() != self.a_id:
                logits[:, 1] = 0.0
                logits[:, 1, 1] = 10.0
        left_length_logits = torch.zeros((batch, 17), device=input_ids.device)
        right_length_logits = torch.zeros((batch, 17), device=input_ids.device)
        return ScaffoldModelOutput(logits, left_length_logits, right_length_logits)


def _flank_output_with_illegal_marginal_maxima() -> ScaffoldModelOutput:
    token_logits = torch.zeros((1, 4, 4))
    left_length_logits = torch.full((1, 11), -100.0)
    right_length_logits = torch.full((1, 11), -100.0)
    left_length_logits[0, 0] = 100.0
    right_length_logits[0, 0] = 100.0
    left_length_logits[0, 2] = 90.0
    right_length_logits[0, 2] = 90.0
    return ScaffoldModelOutput(token_logits, left_length_logits, right_length_logits)


def test_select_flank_lengths_masks_illegal_joint_marginal_maxima():
    left_length, right_length = select_flank_lengths(
        _flank_output_with_illegal_marginal_maxima(),
        motif_length=4,
        max_length=10,
        generator=torch.Generator().manual_seed(7),
        sample=False,
        min_scaffold_length=4,
        min_flank_length=2,
    )

    assert (left_length, right_length) == (2, 2)


def test_select_flank_lengths_sampling_never_returns_illegal_joint_pair():
    selections = {
        select_flank_lengths(
            _flank_output_with_illegal_marginal_maxima(),
            motif_length=4,
            max_length=10,
            generator=torch.Generator().manual_seed(seed),
            sample=True,
            min_scaffold_length=4,
            min_flank_length=2,
        )
        for seed in range(20)
    }

    assert selections <= {(2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (4, 2)}


def test_select_flank_lengths_applies_minimum_to_total_canvas_length():
    token_logits = torch.zeros((1, 20, 4))
    left_length_logits = torch.full((1, 31), -100.0)
    right_length_logits = torch.full((1, 31), -100.0)
    left_length_logits[0, 2] = 100.0
    right_length_logits[0, 3] = 100.0

    left_length, right_length = select_flank_lengths(
        ScaffoldModelOutput(token_logits, left_length_logits, right_length_logits),
        motif_length=20,
        max_length=30,
        generator=torch.Generator().manual_seed(7),
        sample=False,
        min_scaffold_length=25,
        min_flank_length=2,
    )

    assert (left_length, right_length) == (2, 3)


def test_select_flank_lengths_rejects_pairs_above_requested_maximum():
    token_logits = torch.zeros((1, 4, 4))
    left_length_logits = torch.full((1, 11), -100.0)
    right_length_logits = torch.full((1, 11), -100.0)
    left_length_logits[0, 2] = 100.0
    right_length_logits[0, 9] = 100.0
    right_length_logits[0, 4] = 90.0

    left_length, right_length = select_flank_lengths(
        ScaffoldModelOutput(token_logits, left_length_logits, right_length_logits),
        motif_length=4,
        max_length=10,
        generator=torch.Generator().manual_seed(7),
        sample=False,
        min_scaffold_length=1,
        min_flank_length=2,
    )

    assert (left_length, right_length) == (2, 4)


def test_select_flank_lengths_reports_when_no_pair_is_legal():
    with pytest.raises(ValueError, match="motif cannot fit"):
        select_flank_lengths(
            _flank_output_with_illegal_marginal_maxima(),
            motif_length=20,
            max_length=30,
            generator=torch.Generator().manual_seed(7),
            sample=False,
            min_scaffold_length=31,
            min_flank_length=2,
        )


def test_iterative_denoise_is_reproducible_and_preserves_motif():
    tokenizer = RnaTokenizer()
    model = DeterministicScaffoldModel()
    settings = DecodingSettings(denoise_steps=4, temperature=1.0, top_k=1, top_p=1.0)

    def decode():
        return iterative_denoise(
            model,
            tokenizer,
            motif="GCGG",
            total_length=10,
            motif_start=3,
            settings=settings,
            generator=torch.Generator().manual_seed(11),
            device="cpu",
        )

    first = decode()
    second = decode()

    assert first == second
    assert first.sequence[3:7] == "GCGG"
    assert set(first.sequence) <= set("AUCG")
    assert first.unresolved_counts[-1] == 0
    assert all(later < earlier for earlier, later in zip(first.unresolved_counts, first.unresolved_counts[1:]))
    assert torch.isfinite(torch.tensor(first.normalized_log_probability))


def test_iterative_denoise_remasks_low_confidence_tokens_and_scores_final_canvas():
    """Changing a filled token back to MASK must allow context to correct it."""
    tokenizer = RnaTokenizer()
    model = ContextCorrectingScaffoldModel(
        mask_id=tokenizer.token_to_id[tokenizer.special.mask],
        a_id=tokenizer.token_to_id["A"],
    )

    decoded = iterative_denoise(
        model,
        tokenizer,
        motif="GC",
        total_length=6,
        motif_start=2,
        settings=DecodingSettings(
            denoise_steps=3,
            temperature=1.0,
            top_p=1.0,
            mask_schedule="linear",
        ),
        generator=torch.Generator().manual_seed(19),
        device="cpu",
    )

    mask_id = tokenizer.token_to_id[tokenizer.special.mask]
    motif_ids = torch.tensor(tokenizer.encode("GC"))
    assert decoded.sequence[1] == "C"
    assert decoded.masked_counts == (4, 3, 2, 0)
    assert all(later < earlier for earlier, later in zip(decoded.masked_counts, decoded.masked_counts[1:]))
    assert set(decoded.sequence) <= set("AUCG")
    assert len(model.seen_inputs) == 4
    assert model.seen_inputs[1][0, 1].item() == mask_id
    assert all(torch.equal(canvas[0, 2:4], motif_ids) for canvas in model.seen_inputs)
    assert decoded.normalized_log_probability == pytest.approx(-math.log(4), abs=1e-6)
