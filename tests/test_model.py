import pytest
import torch

pytest.importorskip("lightning.pytorch")

from rna_scaffold.lightning_module import RnaScaffoldLitModule, compact_fixed_motifs
from rna_scaffold.tokenizer import RnaTokenizer


def test_lightning_module_returns_finite_joint_denoising_loss():
    tokenizer = RnaTokenizer()
    model = RnaScaffoldLitModule(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        max_length=8,
        lr=1e-3,
    )
    batch = {
        "input_ids": torch.tensor([[3, 8, 11, 3, 0, 0, 0, 0]]),
        "target_base_ids": torch.tensor([[0, 0, 3, 1, -100, -100, -100, -100]]),
        "fixed_mask": torch.tensor([[False, True, True, False, False, False, False, False]]),
        "attention_mask": torch.tensor([[True, True, True, True, False, False, False, False]]),
        "target_length": torch.tensor([4]),
        "motif_start": torch.tensor([1]),
    }

    output = model.training_step(batch, batch_idx=0)

    assert set(output) >= {"loss", "base_loss", "length_loss", "position_loss", "token_accuracy"}
    assert torch.isfinite(output["loss"])


def test_compact_fixed_motifs_removes_flank_length_and_position_leakage():
    input_ids = torch.tensor(
        [[3, 8, 11, 10, 3, 3], [3, 3, 9, 8, 11, 3]]
    )
    fixed_mask = torch.tensor(
        [
            [False, True, True, True, False, False],
            [False, False, True, True, True, False],
        ]
    )

    motifs, attention = compact_fixed_motifs(input_ids, fixed_mask, pad_token_id=0)

    assert motifs.tolist() == [[8, 11, 10], [9, 8, 11]]
    assert attention.tolist() == [[True, True, True], [True, True, True]]
