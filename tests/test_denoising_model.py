import torch

from rna_scaffold.model import (
    MotifDenoisingTransformer,
    ScaffoldModelOutput,
    compute_denoising_losses,
    restore_fixed_tokens,
)


def test_model_outputs_tokens_lengths_positions_and_confidence():
    model = MotifDenoisingTransformer(
        vocab_size=12,
        pad_token_id=0,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        max_length=64,
    ).eval()
    input_ids = torch.tensor([[3, 3, 8, 11, 3, 3]])
    attention_mask = torch.ones(1, 6, dtype=torch.bool)

    output = model(input_ids=input_ids, attention_mask=attention_mask)

    assert output.token_logits.shape == (1, 6, 4)
    assert output.length_logits.shape == (1, 65)
    assert output.position_logits.shape == (1, 64)
    assert output.confidence.shape == (1, 6)


def test_restore_fixed_tokens_never_changes_motif():
    original = torch.tensor([[3, 8, 11, 3]])
    proposed = torch.tensor([[9, 9, 9, 9]])
    fixed = torch.tensor([[False, True, True, False]])

    restored = restore_fixed_tokens(proposed, original, fixed)

    assert restored.tolist() == [[9, 8, 11, 9]]


def test_padding_tokens_do_not_change_valid_position_outputs():
    torch.manual_seed(3)
    model = MotifDenoisingTransformer(
        vocab_size=12,
        pad_token_id=0,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        max_length=16,
    ).eval()
    short = model(
        input_ids=torch.tensor([[3, 8, 11, 3]]),
        attention_mask=torch.tensor([[True, True, True, True]]),
    )
    padded = model(
        input_ids=torch.tensor([[3, 8, 11, 3, 0, 0]]),
        attention_mask=torch.tensor([[True, True, True, True, False, False]]),
    )

    assert torch.allclose(short.token_logits, padded.token_logits[:, :4], atol=1e-6)
    assert torch.allclose(short.length_logits, padded.length_logits, atol=1e-6)


def test_denoising_loss_excludes_fixed_motif_positions():
    output = ScaffoldModelOutput(
        token_logits=torch.tensor([[[8.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0]]]),
        length_logits=torch.zeros(1, 9),
        position_logits=torch.zeros(1, 8),
        confidence=torch.ones(1, 2),
    )
    common = dict(
        output=output,
        fixed_mask=torch.tensor([[False, True]]),
        attention_mask=torch.tensor([[True, True]]),
        target_length=torch.tensor([2]),
        motif_start=torch.tensor([1]),
    )

    first = compute_denoising_losses(target_base_ids=torch.tensor([[0, 0]]), **common)
    changed_fixed_target = compute_denoising_losses(target_base_ids=torch.tensor([[0, 3]]), **common)

    assert torch.allclose(first.base_loss, changed_fixed_target.base_loss)
    assert torch.isfinite(first.total_loss)
