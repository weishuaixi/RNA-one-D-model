import torch
import pytest

pytest.importorskip("lightning.pytorch")
from rna_scaffold.lightning_module import RnaScaffoldLitModule
from rna_scaffold.tokenizer import RnaTokenizer


def test_lightning_module_forward_returns_loss_for_teacher_forcing_batch():
    tokenizer = RnaTokenizer()
    model = RnaScaffoldLitModule(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        lr=1e-3,
    )
    batch = {
        "input_ids": torch.tensor([[tokenizer.bos_token_id, tokenizer.token_to_id["A"], tokenizer.eos_token_id]]),
        "labels": torch.tensor(
            [[tokenizer.bos_token_id, tokenizer.token_to_id["C"], tokenizer.eos_token_id]]
        ),
    }

    output = model.training_step(batch, batch_idx=0)

    assert "loss" in output
    assert torch.isfinite(output["loss"])
