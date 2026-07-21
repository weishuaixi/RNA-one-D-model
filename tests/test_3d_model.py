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
    torsion_angle_loss,
)
from rna_scaffold_3d.rhofold import RhoFoldConfig, RhoFoldModel
from rna_scaffold_3d.rna_atoms import RNA_ATOM_NAMES, RNA_NUM_ATOMS


def test_rhofold_model_returns_structure_sequence_embedding_and_confidence():
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
    assert output["plddt"].shape == (1, 5)
    assert output["sequence_logits"].shape == (1, 5, config.vocab_size)
    assert output["sequence_embedding"].shape == (1, 5, config.d_model)
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


def test_structure_loss_updates_trainable_sequence_embedding():
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
    output = model(torch.tensor([[1, 2, 3, 4]]), return_aux=True)

    output["coords"].pow(2).mean().backward()

    gradient = model.seq_embedder.embedding.weight.grad
    assert gradient is not None
    assert gradient.abs().sum().item() > 0


def test_3d_loss_backpropagates_through_soft_sequence_generation():
    config = RhoFoldConfig(
        d_model=32,
        pair_dim=16,
        msa_dim=24,
        nhead=4,
        num_e2e_layers=2,
        num_structure_layers=1,
        dim_feedforward=64,
        num_distance_bins=8,
        recycle_iters=1,
    )
    model = RhoFoldModel(config)
    input_ids = torch.tensor([[1, 5, 5, 4]])
    output = model(input_ids, return_aux=True)

    output["coords"].pow(2).mean().backward()

    generator_gradient = model.sequence_head[-1].weight.grad
    assert generator_gradient is not None
    assert generator_gradient.abs().sum().item() > 0


def test_known_sequence_positions_are_not_replaced_by_soft_predictions():
    config = RhoFoldConfig(
        d_model=16,
        pair_dim=8,
        msa_dim=8,
        nhead=4,
        num_e2e_layers=1,
        num_structure_layers=1,
        dim_feedforward=32,
    )
    model = RhoFoldModel(config)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    sequence_embedding = model.seq_embedder(input_ids)
    logits = torch.randn((1, 4, config.vocab_size))

    injected = model.seq_embedder.inject_predicted_bases(sequence_embedding, logits, input_ids)

    assert torch.equal(injected, sequence_embedding)


def test_model_learns_relative_sequence_and_structure_loss_weights():
    model = RhoFoldModel(
        RhoFoldConfig(
            d_model=16,
            pair_dim=8,
            msa_dim=8,
            nhead=4,
            num_e2e_layers=1,
            num_structure_layers=1,
            dim_feedforward=32,
            sequence_loss_initial_weight=0.1,
        )
    )
    structure_weight, sequence_weight = model.learned_task_weights()
    structure_loss = torch.tensor(2.0, requires_grad=True)
    sequence_loss = torch.tensor(1.5, requires_grad=True)

    combined = model.combine_task_losses(structure_loss, sequence_loss)
    combined.backward()

    assert structure_weight.item() == pytest.approx(1.0)
    assert sequence_weight.item() == pytest.approx(0.1)
    assert model.task_log_variances.grad is not None
    assert model.task_log_variances.grad.abs().sum().item() > 0
