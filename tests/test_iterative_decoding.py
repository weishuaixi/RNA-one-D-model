import torch

from rna_scaffold.decoding import DecodingSettings, iterative_denoise, select_length_position
from rna_scaffold.model import ScaffoldModelOutput
from rna_scaffold.tokenizer import RnaTokenizer


class DeterministicScaffoldModel(torch.nn.Module):
    max_length = 16

    def forward(self, input_ids, attention_mask=None):
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 4), -10.0, device=input_ids.device)
        preferred = torch.arange(length, device=input_ids.device) % 4
        logits.scatter_(2, preferred.view(1, length, 1).expand(batch, -1, -1), 10.0)
        length_logits = torch.full((batch, 17), -100.0, device=input_ids.device)
        length_logits[:, 10] = 10.0
        position_logits = torch.full((batch, 16), -100.0, device=input_ids.device)
        position_logits[:, 3] = 10.0
        return ScaffoldModelOutput(logits, length_logits, position_logits)


def test_select_length_position_masks_impossible_pairs():
    output = DeterministicScaffoldModel()(
        torch.tensor([[3, 3, 3, 3]]), torch.ones(1, 4, dtype=torch.bool)
    )

    total_length, motif_start = select_length_position(
        output,
        motif_length=8,
        max_length=9,
        generator=torch.Generator().manual_seed(7),
        sample=False,
        min_scaffold_length=1,
        min_flank_length=0,
    )

    assert total_length == 9
    assert motif_start == 0


def test_select_length_position_enforces_scaffold_and_bilateral_flanks():
    output = DeterministicScaffoldModel()(
        torch.tensor([[3, 3, 3, 3]]), torch.ones(1, 4, dtype=torch.bool)
    )

    total_length, motif_start = select_length_position(
        output,
        motif_length=4,
        max_length=8,
        generator=torch.Generator().manual_seed(7),
        sample=False,
        min_scaffold_length=4,
        min_flank_length=2,
    )

    assert (total_length, motif_start) == (8, 2)


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
