import torch
import pytest

from rna_scaffold_3d.losses import (
    bond_angle_loss,
    bond_length_loss,
    masked_coordinate_huber,
    masked_coordinate_mse,
    masked_pairwise_distance_mse,
    pair_distance_cross_entropy,
    plddt_confidence_loss,
    secondary_logits_bce_loss,
    secondary_structure_pair_loss,
    torsion_angle_loss,
)
from rna_scaffold_3d.rhofold import RhoFoldConfig, RhoFoldModel
from rna_scaffold_3d.rna_atoms import RNA_ATOM_NAMES, RNA_NUM_ATOMS


def test_rhofold_model_returns_coords_distogram_secondary_and_confidence():
    config = RhoFoldConfig(
        d_model=32,
        pair_dim=16,
        msa_dim=24,
        nhead=4,
        num_e2e_layers=1,
        num_structure_layers=1,
        dim_feedforward=64,
        num_distance_bins=8,
        recycle_iters=1,
    )
    model = RhoFoldModel(config)
    input_ids = torch.tensor([[1, 2, 3, 4, 0]])
    msa_ids = torch.tensor([[[1, 2, 3, 4, 0], [1, 2, 3, 2, 0]]])
    padding_mask = torch.tensor([[False, False, False, False, True]])
    msa_mask = torch.tensor([[[False, False, False, False, True], [False, False, False, False, True]]])

    output = model(
        input_ids=input_ids,
        padding_mask=padding_mask,
        msa_ids=msa_ids,
        msa_mask=msa_mask,
        return_aux=True,
    )

    assert output["coords"].shape == (1, 5, RNA_NUM_ATOMS, 3)
    assert output["pair_distance_logits"].shape == (1, 5, 5, 8)
    assert output["secondary_logits"].shape == (1, 5, 5)
    assert output["plddt"].shape == (1, 5)
    assert torch.all(output["plddt"] >= 0)
    assert torch.all(output["plddt"] <= 100)


def test_masked_coordinate_losses_ignore_invalid_positions():
    pred = torch.tensor([[[1.0, 1.0, 1.0], [100.0, 100.0, 100.0]]])
    target = torch.tensor([[[2.0, 1.0, 1.0], [0.0, 0.0, 0.0]]])
    mask = torch.tensor([[True, False]])

    assert masked_coordinate_mse(pred, target, mask).item() == pytest.approx(1.0 / 3.0)
    assert masked_pairwise_distance_mse(pred, target, mask).item() == 0.0


def test_masked_coordinate_huber_is_less_sensitive_to_large_outliers():
    pred = torch.tensor([[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]]])
    target = torch.zeros_like(pred)
    mask = torch.tensor([[True, True]])

    assert masked_coordinate_huber(pred, target, mask, beta=1.0).item() < masked_coordinate_mse(pred, target, mask).item()


def test_plddt_confidence_loss_rewards_accurate_residue_coordinates():
    pred = torch.zeros((1, 2, RNA_NUM_ATOMS, 3))
    target = torch.zeros_like(pred)
    mask = torch.ones((1, 2, RNA_NUM_ATOMS), dtype=torch.bool)
    target[:, 1] = 10.0
    predicted_plddt = torch.tensor([[95.0, 95.0]], requires_grad=True)

    loss = plddt_confidence_loss(predicted_plddt, pred, target, mask)
    loss.backward()

    assert loss.item() > 0
    assert predicted_plddt.grad is not None
    assert predicted_plddt.grad.abs().sum().item() > 0


def test_pair_distance_cross_entropy_uses_target_distance_bins():
    logits = torch.zeros((1, 2, 2, 4))
    coords = torch.zeros((1, 2, 4, 3))
    mask = torch.zeros((1, 2, 4), dtype=torch.bool)
    coords[0, 0, 0] = torch.tensor([0.0, 0.0, 0.0])
    coords[0, 1, 0] = torch.tensor([4.0, 0.0, 0.0])
    mask[0, :, 0] = True

    loss = pair_distance_cross_entropy(logits, coords, mask, bin_edges=torch.tensor([2.0, 6.0, 10.0]))

    assert loss.item() > 0


def test_geometry_losses_are_zero_for_valid_simple_geometry():
    atom_count = len(RNA_ATOM_NAMES)
    coords = torch.zeros((1, 2, atom_count, 3))
    mask = torch.zeros((1, 2, atom_count), dtype=torch.bool)
    p = RNA_ATOM_NAMES.index("P")
    o5 = RNA_ATOM_NAMES.index("O5'")
    c5 = RNA_ATOM_NAMES.index("C5'")
    c4 = RNA_ATOM_NAMES.index("C4'")
    c3 = RNA_ATOM_NAMES.index("C3'")
    o3 = RNA_ATOM_NAMES.index("O3'")
    coords[0, 0, p] = torch.tensor([0.0, 0.0, 0.0])
    coords[0, 0, o5] = torch.tensor([1.6, 0.0, 0.0])
    coords[0, 0, c5] = torch.tensor([3.0, 0.0, 0.0])
    coords[0, 0, c4] = torch.tensor([4.5, 0.0, 0.0])
    coords[0, 0, c3] = torch.tensor([6.0, 0.0, 0.0])
    coords[0, 0, o3] = torch.tensor([7.4, 0.0, 0.0])
    coords[0, 1, p] = torch.tensor([8.9, 0.0, 0.0])
    mask[0, 0, [p, o5, c5, c4, c3, o3]] = True

    assert bond_length_loss(coords, mask).item() == pytest.approx(0.0)
    assert bond_angle_loss(coords, mask).item() >= 0.0
    assert torsion_angle_loss(coords, mask).item() >= 0.0


def test_secondary_structure_pair_loss_penalizes_far_complementary_pairs():
    atom_count = len(RNA_ATOM_NAMES)
    coords = torch.zeros((1, 4, atom_count, 3))
    mask = torch.zeros((1, 4, atom_count), dtype=torch.bool)
    c4 = RNA_ATOM_NAMES.index("C4'")
    coords[0, 0, c4] = torch.tensor([0.0, 0.0, 0.0])
    coords[0, 3, c4] = torch.tensor([20.0, 0.0, 0.0])
    mask[0, [0, 3], c4] = True
    input_ids = torch.tensor([[1, 3, 3, 2]])  # A C C U, ends can pair.

    loss = secondary_structure_pair_loss(coords, mask, input_ids)

    assert loss.item() > 0


def test_secondary_logits_bce_loss_trains_pairing_head_from_sequence_pairs():
    logits = torch.zeros((1, 4, 4), requires_grad=True)
    input_ids = torch.tensor([[1, 3, 3, 2]])  # A C C U
    padding_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    loss = secondary_logits_bce_loss(logits, input_ids, padding_mask)
    loss.backward()

    assert loss.item() > 0
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0
