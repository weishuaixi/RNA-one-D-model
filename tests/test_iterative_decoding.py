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
