import math

import torch
import pytest

from rna_scaffold_3d.losses import (
    base_orientation_coordinate_loss,
    _dihedral,
    frame_aligned_point_error,
    inter_residue_geometry_loss,
    kabsch_aligned_coordinate_loss,
    local_distance_difference_loss,
    soft_lddt_loss,
    bond_angle_loss,
    bond_length_loss,
    masked_coordinate_huber,
    masked_coordinate_mse,
    masked_pairwise_distance_mse,
    pair_distance_cross_entropy,
    pair_orientation_cross_entropy,
    plddt_confidence_loss,
    sugar_pucker_phase,
    sugar_pucker_coordinate_loss,
    sugar_pucker_phase_loss,
    steric_clash_loss,
    torsion_angle_loss,
    torsion_parameter_loss,
)
from rna_scaffold_3d.geometry import (
    apply_random_rigid_augmentation,
    build_residue_frames,
    rotation_6d_to_matrix,
)
from rna_scaffold_3d.internal_coords import build_rna_backbone
from rna_scaffold_3d.metrics import batch_structure_metrics
from rna_scaffold_3d.rhofold import (
    EquivariantInternalCoordinateRefinement,
    InvariantPointAttention,
    OuterProductMean,
    PairAttentionPooling,
    PairBiasedSelfAttention,
    PairTransition,
    RecyclingEmbedder,
    RhoFoldConfig,
    RhoFoldModel,
    TriangleAttention,
    TriangleMultiplicativeUpdate,
)
from rna_scaffold_3d.rna_atoms import (
    RNA_ATOM_NAMES,
    RNA_ATOM_TO_INDEX,
    RNA_NUM_ATOMS,
    chemical_bond_adjacency,
    chemical_atom_mask,
)


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


def test_sequence_objective_backpropagates_through_final_trunk_block():
    torch.manual_seed(101)
    config = RhoFoldConfig(
        d_model=24,
        pair_dim=12,
        msa_dim=12,
        nhead=4,
        pair_heads=3,
        num_e2e_layers=2,
        num_structure_layers=1,
        dim_feedforward=48,
        dropout=0.0,
        recycle_iters=1,
    )
    model = RhoFoldModel(config)
    input_ids = torch.tensor([[5, 2, 3, 4]])

    output = model(input_ids=input_ids, return_aux=True)
    output["sequence_logits"].square().mean().backward()

    final_transition = model.e2eformer[-1].seq_transition[1]
    assert final_transition.weight.grad is not None
    assert final_transition.weight.grad.abs().sum() > 0


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


def test_coordinate_losses_weight_mixed_length_rnas_equally():
    pred = torch.zeros((2, 3, 3))
    target = torch.zeros_like(pred)
    mask = torch.zeros((2, 3), dtype=torch.bool)
    mask[0, 0] = True
    mask[1] = True
    pred[:, 0, 0] = 1.0

    for loss_fn in (masked_coordinate_mse, masked_coordinate_huber):
        batch_loss = loss_fn(pred, target, mask)
        separate_mean = (
            loss_fn(pred[:1], target[:1], mask[:1])
            + loss_fn(pred[1:], target[1:], mask[1:])
        ) / 2.0
        assert batch_loss.item() == pytest.approx(
            separate_mean.item(), rel=1e-6
        )


def test_empty_label_example_does_not_dilute_coordinate_loss():
    pred = torch.zeros((2, 1, 3))
    target = torch.zeros_like(pred)
    mask = torch.tensor([[True], [False]])
    pred[0, 0, 0] = 1.0

    assert masked_coordinate_mse(pred, target, mask).item() == pytest.approx(
        masked_coordinate_mse(pred[:1], target[:1], mask[:1]).item()
    )


def test_plddt_confidence_loss_rewards_accurate_residue_coordinates():
    pred = torch.zeros((1, 2, RNA_NUM_ATOMS, 3))
    target = torch.zeros_like(pred)
    mask = torch.ones((1, 2, RNA_NUM_ATOMS), dtype=torch.bool)
    target[:, 1, :, 0] = 10.0
    predicted_plddt = torch.tensor([[95.0, 95.0]], requires_grad=True)

    loss = plddt_confidence_loss(predicted_plddt, pred, target, mask)
    loss.backward()

    assert loss.item() > 0
    assert predicted_plddt.grad is not None
    assert predicted_plddt.grad.abs().sum().item() > 0


def test_structure_metrics_report_valid_example_counts():
    coords = torch.zeros((2, 3, RNA_NUM_ATOMS, 3))
    mask = torch.zeros(coords.shape[:-1], dtype=torch.bool)
    c1 = RNA_ATOM_TO_INDEX["C1'"]
    mask[0, 0, c1] = True
    mask[1, :, c1] = True
    coords[1, :, c1, 0] = torch.tensor([0.0, 5.0, 10.0])

    metrics = batch_structure_metrics(coords, coords, mask)

    assert metrics["kabsch_rmsd_count"].item() == 2.0
    assert metrics["distance_rmsd_count"].item() == 1.0
    assert metrics["c1_lddt_count"].item() == 1.0
    assert metrics["adjacent_c1_mean_count"].item() == 1.0


def test_pair_distance_cross_entropy_uses_target_distance_bins():
    logits = torch.zeros((1, 2, 2, 4))
    coords = torch.zeros((1, 2, RNA_NUM_ATOMS, 3))
    mask = torch.zeros((1, 2, RNA_NUM_ATOMS), dtype=torch.bool)
    c1 = RNA_ATOM_NAMES.index("C1'")
    coords[0, 0, c1] = torch.tensor([0.0, 0.0, 0.0])
    coords[0, 1, c1] = torch.tensor([4.0, 0.0, 0.0])
    mask[0, :, c1] = True

    loss = pair_distance_cross_entropy(logits, coords, mask, bin_edges=torch.tensor([2.0, 6.0, 10.0]))

    assert loss.item() > 0


def test_orientation_and_contact_heads_receive_geometry_gradients():
    torch.manual_seed(2)
    coords = torch.randn(1, 3, RNA_NUM_ATOMS, 3)
    mask = torch.ones(coords.shape[:-1], dtype=torch.bool)
    logits = {
        "omega": torch.zeros(1, 3, 3, 8, requires_grad=True),
        "theta": torch.zeros(1, 3, 3, 8, requires_grad=True),
        "phi": torch.zeros(1, 3, 3, 4, requires_grad=True),
        "contact": torch.zeros(1, 3, 3, 1, requires_grad=True),
    }

    loss = pair_orientation_cross_entropy(logits, coords, mask)
    loss.backward()

    assert loss.item() > 0
    assert all(value.grad is not None and value.grad.abs().sum() > 0 for value in logits.values())


def test_contact_loss_weights_mixed_length_rnas_equally():
    coords = torch.zeros((2, 3, 3))
    coords[:, :, 0] = torch.tensor([0.0, 5.0, 20.0])
    mask = torch.tensor([
        [True, True, False],
        [True, True, True],
    ])
    logits = torch.zeros((2, 3, 3, 1))
    logits[1] = -4.0

    batch_loss = pair_orientation_cross_entropy(
        {"contact": logits}, coords, mask
    )
    separate_mean = (
        pair_orientation_cross_entropy(
            {"contact": logits[:1]}, coords[:1], mask[:1]
        )
        + pair_orientation_cross_entropy(
            {"contact": logits[1:]}, coords[1:], mask[1:]
        )
    ) / 2.0

    assert batch_loss.item() == pytest.approx(
        separate_mean.item(), rel=1e-6
    )


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
    assert torsion_angle_loss(coords, coords, mask).item() >= 0.0


def test_torsion_coordinate_loss_rejects_missing_supervision():
    coords = torch.zeros((1, 2, RNA_NUM_ATOMS, 3))

    with pytest.raises(ValueError, match="requires predicted coordinates"):
        torsion_angle_loss(coords, None, None)


def test_degenerate_geometry_losses_keep_backward_gradients_finite():
    angle_coords = torch.zeros(
        (1, 1, RNA_NUM_ATOMS, 3), requires_grad=True
    )
    angle_mask = torch.zeros(
        angle_coords.shape[:-1], dtype=torch.bool
    )
    for atom, position in (
        ("P", 0.0),
        ("O5'", 1.6),
        ("C5'", 3.0),
    ):
        index = RNA_ATOM_TO_INDEX[atom]
        angle_coords.data[0, 0, index, 0] = position
        angle_mask[0, 0, index] = True
    bond_angle_loss(angle_coords, angle_mask).backward()

    collapsed = torch.zeros((1, 4, 3), requires_grad=True)
    collapsed_target = torch.zeros_like(collapsed)
    collapsed_mask = torch.ones((1, 4), dtype=torch.bool)
    kabsch_aligned_coordinate_loss(
        collapsed, collapsed_target, collapsed_mask
    ).backward()

    torsion_coords = torch.zeros(
        (1, 1, RNA_NUM_ATOMS, 3), requires_grad=True
    )
    torsion_target = torch.zeros_like(torsion_coords)
    torsion_mask = torch.ones(
        torsion_coords.shape[:-1], dtype=torch.bool
    )
    torsion_angle_loss(
        torsion_coords, torsion_target, torsion_mask
    ).backward()

    assert torch.isfinite(angle_coords.grad).all()
    assert torch.isfinite(collapsed.grad).all()
    assert torch.isfinite(torsion_coords.grad).all()


def test_bond_loss_weights_mixed_length_rnas_equally():
    coords = torch.zeros((2, 3, RNA_NUM_ATOMS, 3))
    mask = torch.zeros(coords.shape[:-1], dtype=torch.bool)
    p = RNA_ATOM_TO_INDEX["P"]
    o5 = RNA_ATOM_TO_INDEX["O5'"]
    mask[0, 0, [p, o5]] = True
    mask[1, :, [p, o5]] = True
    coords[..., o5, 0] = 1.6
    coords[:, 0, o5, 0] = 2.6

    batch_loss = bond_length_loss(coords, mask)
    separate_mean = (
        bond_length_loss(coords[:1], mask[:1])
        + bond_length_loss(coords[1:], mask[1:])
    ) / 2.0

    assert batch_loss.item() == pytest.approx(
        separate_mean.item(), rel=1e-6
    )


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


def test_known_bases_use_base_specific_atom_templates():
    model = RhoFoldModel(
        RhoFoldConfig(
            d_model=16,
            pair_dim=8,
            msa_dim=8,
            nhead=4,
            pair_heads=2,
            num_e2e_layers=1,
            num_structure_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            equivariant_layers=0,
        )
    ).eval()

    with torch.no_grad():
        coords = model(torch.tensor([[1, 2]]))

    c1 = RNA_ATOM_TO_INDEX["C1'"]
    n9 = RNA_ATOM_TO_INDEX["N9"]
    o4 = RNA_ATOM_TO_INDEX["O4"]
    assert torch.linalg.vector_norm(coords[0, 0, n9] - coords[0, 0, c1]) > 0.5
    assert torch.linalg.vector_norm(coords[0, 1, n9] - coords[0, 1, c1]) < 1e-5
    assert torch.linalg.vector_norm(coords[0, 0, o4] - coords[0, 0, c1]) < 1e-5
    assert torch.linalg.vector_norm(coords[0, 1, o4] - coords[0, 1, c1]) > 0.5


def test_masked_base_template_mixture_backpropagates_to_sequence_logits():
    model = RhoFoldModel(
        RhoFoldConfig(
            d_model=16,
            pair_dim=8,
            msa_dim=8,
            nhead=4,
            pair_heads=2,
            num_e2e_layers=1,
            num_structure_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            equivariant_layers=0,
        )
    )
    output = model(torch.tensor([[5, 5, 5]]), return_aux=True)
    c1 = RNA_ATOM_TO_INDEX["C1'"]
    n6 = RNA_ATOM_TO_INDEX["N6"]

    (output["coords"][..., n6, :] - output["coords"][..., c1, :]).pow(2).mean().backward()

    gradient = model.sequence_head[-1].weight.grad
    assert gradient is not None
    assert gradient.abs().sum().item() > 0


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


def test_task_weight_clamps_keep_extreme_parameters_finite():
    model = RhoFoldModel(
        RhoFoldConfig(
            d_model=16,
            pair_dim=8,
            msa_dim=8,
            nhead=4,
            num_e2e_layers=1,
            num_structure_layers=1,
            dim_feedforward=32,
        )
    )
    with torch.no_grad():
        model.task_log_variances.copy_(
            torch.tensor([1e9, -1e9])
        )

    weights = model.learned_task_weights()
    combined = model.combine_task_losses(
        torch.tensor(1e6), torch.tensor(1e6)
    )
    combined.backward()

    assert all(torch.isfinite(weight) for weight in weights)
    assert torch.isfinite(combined)
    assert max(weight.item() for weight in weights) <= math.exp(5.0) + 1e-4
    assert model.task_log_variances.grad is not None
    assert torch.isfinite(model.task_log_variances.grad).all()
    assert (model.task_log_variances.grad != 0).all()


def test_task_weight_parameters_outside_bounds_can_recover():
    model = RhoFoldModel(
        RhoFoldConfig(
            d_model=16,
            pair_dim=8,
            msa_dim=8,
            nhead=4,
            num_e2e_layers=1,
            num_structure_layers=1,
            dim_feedforward=32,
        )
    )
    with torch.no_grad():
        model.task_log_variances.copy_(torch.tensor([6.0, -6.0]))
    before = model.task_log_variances.detach().abs().clone()
    optimizer = torch.optim.SGD([model.task_log_variances], lr=0.01)

    optimizer.zero_grad()
    model.combine_task_losses(torch.tensor(1.0), torch.tensor(1.0)).backward()
    optimizer.step()

    after = model.task_log_variances.detach().abs()
    assert torch.all(after < before)


def test_random_rigid_augmentation_preserves_pairwise_distances():
    torch.manual_seed(3)
    coords = torch.randn(2, 4, RNA_NUM_ATOMS, 3)
    mask = torch.ones(coords.shape[:-1], dtype=torch.bool)

    augmented = apply_random_rigid_augmentation(coords, mask)

    before = torch.cdist(coords[:, :, 0], coords[:, :, 0])
    after = torch.cdist(augmented[:, :, 0], augmented[:, :, 0])
    assert torch.allclose(before, after, atol=1e-4)


def test_kabsch_and_fape_are_invariant_to_global_rigid_transform():
    torch.manual_seed(5)
    target = torch.randn(1, 4, RNA_NUM_ATOMS, 3)
    mask = torch.ones(target.shape[:-1], dtype=torch.bool)
    pred = apply_random_rigid_augmentation(target, mask)

    assert kabsch_aligned_coordinate_loss(pred, target, mask).item() < 1e-4
    assert frame_aligned_point_error(pred, target, mask).item() < 1e-4
    assert local_distance_difference_loss(pred, target, mask).item() < 1e-4
    assert soft_lddt_loss(pred, target, mask).item() < 1e-4


def test_kabsch_loss_promotes_bfloat16_inputs_for_svd_and_backward():
    torch.manual_seed(105)
    target = torch.randn(1, 4, RNA_NUM_ATOMS, 3, dtype=torch.bfloat16)
    pred = (target.float() + 0.1 * torch.randn_like(target.float())).to(torch.bfloat16)
    pred.requires_grad_()
    mask = torch.ones(target.shape[:-1], dtype=torch.bool)

    loss = kabsch_aligned_coordinate_loss(pred, target, mask)

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_soft_lddt_loss_penalizes_local_distance_errors_and_is_differentiable():
    target = torch.tensor(
        [[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]]]
    )
    pred = target.clone()
    pred[:, 1, 0] += 3.0
    pred.requires_grad_()
    mask = torch.ones(1, 3, dtype=torch.bool)

    loss = soft_lddt_loss(pred, target, mask)
    loss.backward()

    assert loss.item() > 0.1
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0.0


def test_local_distance_loss_weights_structures_not_pair_counts():
    torch.manual_seed(51)
    target = torch.randn(2, 8, 3)
    pred = target.clone()
    pred[0, :3] += torch.tensor([2.0, 0.0, 0.0])
    pred[1, :8] += torch.linspace(0.0, 1.0, 8).view(-1, 1)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[0, :3] = True
    mask[1, :8] = True

    batch_loss = local_distance_difference_loss(pred, target, mask)
    individual = torch.stack(
        [
            local_distance_difference_loss(
                pred[index:index + 1],
                target[index:index + 1],
                mask[index:index + 1],
            )
            for index in range(2)
        ]
    ).mean()

    assert torch.allclose(batch_loss, individual, atol=1e-6)


def test_all_atom_fape_detects_non_representative_atom_error():
    torch.manual_seed(5)
    target = torch.randn(1, 4, RNA_NUM_ATOMS, 3)
    mask = torch.ones(target.shape[:-1], dtype=torch.bool)
    pred = target.clone()
    op1 = RNA_ATOM_NAMES.index("OP1")
    pred[:, :, op1, 0] += 3.0

    loss = frame_aligned_point_error(pred, target, mask)

    assert loss.item() > 0.01


def test_fape_penalizes_collapsed_predicted_frames_and_can_recover():
    torch.manual_seed(52)
    target = torch.randn(1, 4, RNA_NUM_ATOMS, 3)
    mask = torch.ones(target.shape[:-1], dtype=torch.bool)
    pred = torch.zeros_like(target, requires_grad=True)

    correct_loss = frame_aligned_point_error(target, target, mask)
    collapsed_loss = frame_aligned_point_error(pred, target, mask)
    collapsed_loss.backward()

    assert correct_loss.item() < 1e-6
    assert collapsed_loss.item() > 0.05
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.norm().item() > 0.0


def test_frame_and_aligned_losses_average_examples_not_valid_atom_counts():
    torch.manual_seed(51)
    target = torch.randn(2, 6, RNA_NUM_ATOMS, 3)
    mask = torch.ones(target.shape[:-1], dtype=torch.bool)
    mask[0, 2:] = False
    pred = target.clone()
    op1 = RNA_ATOM_TO_INDEX["OP1"]
    pred[0, :2, op1, 0] += 1.0
    pred[1, :, op1, 0] += 5.0

    batch_fape = frame_aligned_point_error(pred, target, mask)
    expected_fape = torch.stack(
        [
            frame_aligned_point_error(pred[index:index + 1], target[index:index + 1], mask[index:index + 1])
            for index in range(2)
        ]
    ).mean()
    batch_aligned = kabsch_aligned_coordinate_loss(pred, target, mask)
    expected_aligned = torch.stack(
        [
            kabsch_aligned_coordinate_loss(pred[index:index + 1], target[index:index + 1], mask[index:index + 1])
            for index in range(2)
        ]
    ).mean()

    assert torch.allclose(batch_fape, expected_fape, atol=1e-6)
    assert torch.allclose(batch_aligned, expected_aligned, atol=1e-6)


def test_chemical_atom_mask_uses_sequence_identity_not_observation_coverage():
    mask = chemical_atom_mask(torch.tensor([[1, 2, 3, 4, 0, 5]]))

    assert mask[0, 0, RNA_ATOM_TO_INDEX["N9"]]
    assert not mask[0, 0, RNA_ATOM_TO_INDEX["O4"]]
    assert mask[0, 1, RNA_ATOM_TO_INDEX["O4"]]
    assert not mask[0, 1, RNA_ATOM_TO_INDEX["N9"]]
    assert mask[0, 2, RNA_ATOM_TO_INDEX["N4"]]
    assert mask[0, 3, RNA_ATOM_TO_INDEX["O6"]]
    assert not mask[0, 4:].any()


def test_orientation_loss_is_invariant_to_global_rigid_transform():
    torch.manual_seed(6)
    target = torch.randn(1, 5, RNA_NUM_ATOMS, 3)
    mask = torch.ones(target.shape[:-1], dtype=torch.bool)
    logits = {
        "omega": torch.randn(1, 5, 5, 12),
        "theta": torch.randn(1, 5, 5, 12),
        "phi": torch.randn(1, 5, 5, 6),
        "contact": torch.randn(1, 5, 5, 1),
    }
    transformed = apply_random_rigid_augmentation(target, mask)

    original_loss = pair_orientation_cross_entropy(logits, target, mask)
    transformed_loss = pair_orientation_cross_entropy(logits, transformed, mask)

    assert torch.allclose(original_loss, transformed_loss, atol=1e-6)


def test_base_orientation_coordinate_loss_is_rigid_invariant_and_differentiable():
    model = RhoFoldModel(
        RhoFoldConfig(
            d_model=16,
            pair_dim=8,
            msa_dim=8,
            nhead=4,
            pair_heads=2,
            num_e2e_layers=1,
            num_structure_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            equivariant_layers=0,
        )
    ).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        target = model(input_ids)[0].unsqueeze(0)
    mask = chemical_atom_mask(input_ids)
    transformed = apply_random_rigid_augmentation(target, mask)
    assert base_orientation_coordinate_loss(
        transformed, target, mask, input_ids
    ).item() < 1e-6

    pred = target.clone().requires_grad_(True)
    displaced = torch.zeros_like(pred)
    displaced[0, 0, RNA_ATOM_TO_INDEX["C4"]] = torch.tensor(
        [0.6, -0.8, 1.0]
    )
    loss = base_orientation_coordinate_loss(
        pred + displaced, target, mask, input_ids
    )
    loss.backward()

    assert loss.item() > 0.0
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0.0


def test_base_orientation_coordinate_loss_masks_missing_base_frame():
    input_ids = torch.tensor([[1]])
    target = torch.randn(1, 1, RNA_NUM_ATOMS, 3)
    mask = chemical_atom_mask(input_ids)
    mask[..., RNA_ATOM_TO_INDEX["C4"]] = False
    pred = target.clone()
    pred[..., RNA_ATOM_TO_INDEX["C4"], :] += 10.0

    loss = base_orientation_coordinate_loss(
        pred, target, mask, input_ids
    )

    assert loss.item() == 0.0


def test_direct_base_orientation_loss_reaches_orientation_head():
    model = RhoFoldModel(
        RhoFoldConfig(
            d_model=16,
            pair_dim=8,
            msa_dim=8,
            nhead=4,
            pair_heads=2,
            num_e2e_layers=1,
            num_structure_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            equivariant_layers=0,
        )
    )
    input_ids = torch.tensor([[1, 2, 3, 4]])
    output = model(input_ids, return_aux=True)
    target = output["coords"].detach().clone()
    target[0, 0, RNA_ATOM_TO_INDEX["C4"]] += torch.tensor(
        [0.5, -0.7, 0.9]
    )
    loss = base_orientation_coordinate_loss(
        output["coords"],
        target,
        chemical_atom_mask(input_ids),
        input_ids,
    )
    loss.backward()

    gradient = model.structure_module.base_orientation_head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0.0


def test_degenerate_residue_atoms_do_not_define_valid_frames():
    coords = torch.zeros(1, 2, RNA_NUM_ATOMS, 3)
    mask = torch.ones(coords.shape[:-1], dtype=torch.bool)

    _, _, frame_mask = build_residue_frames(coords, mask)

    assert not frame_mask.any()


def test_sequence_aware_frames_do_not_use_purine_ring_n1_as_glycosidic_atom():
    coords = torch.zeros(1, 2, RNA_NUM_ATOMS, 3)
    mask = torch.zeros(coords.shape[:-1], dtype=torch.bool)
    c4 = RNA_ATOM_TO_INDEX["C4'"]
    c1 = RNA_ATOM_TO_INDEX["C1'"]
    n1 = RNA_ATOM_TO_INDEX["N1"]
    coords[0, :, c4] = torch.tensor([0.0, 0.0, 0.0])
    coords[0, :, c1] = torch.tensor([1.0, 0.0, 0.0])
    coords[0, :, n1] = torch.tensor([1.0, 1.0, 0.0])
    mask[0, :, [c4, c1, n1]] = True

    _, _, valid = build_residue_frames(
        coords,
        mask,
        input_ids=torch.tensor([[1, 2]]),
    )

    assert not valid[0, 0]  # A requires the missing N9.
    assert valid[0, 1]      # U correctly uses N1.


def test_torsion_loss_selects_exactly_one_base_specific_chi():
    torch.manual_seed(71)
    target = torch.randn(1, 3, RNA_NUM_ATOMS, 3)
    mask = torch.ones(target.shape[:-1], dtype=torch.bool)
    pred = target.clone()
    pred[..., RNA_ATOM_TO_INDEX["N1"], 2] += 1.5

    purine_loss = torsion_angle_loss(
        pred, target, mask, torch.tensor([[1, 1, 1]])
    )
    pyrimidine_loss = torsion_angle_loss(
        pred, target, mask, torch.tensor([[2, 2, 2]])
    )

    assert purine_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert pyrimidine_loss.item() > 0.01


def test_periodic_torsion_loss_compares_prediction_to_target():
    torch.manual_seed(7)
    target = torch.randn(1, 5, RNA_NUM_ATOMS, 3)
    mask = torch.ones(target.shape[:-1], dtype=torch.bool)
    same_loss = torsion_angle_loss(target.clone(), target, mask)
    perturbed = target.clone()
    perturbed[:, :, RNA_ATOM_NAMES.index("C4'"), 2] += 2.0

    assert same_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert torsion_angle_loss(perturbed, target, mask).item() > same_loss.item()


def test_internal_coordinate_backbone_torsions_round_trip_to_direct_targets():
    torch.manual_seed(61)
    angles = (torch.rand(2, 7, 7) * 2.0 - 1.0) * 3.0
    torsions = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)
    padding_mask = torch.zeros(2, 7, dtype=torch.bool)
    rotation = torch.eye(3).expand(2, 3, 3)
    backbone, _, _ = build_rna_backbone(torsions, padding_mask, rotation)
    coords = torch.zeros(2, 7, RNA_NUM_ATOMS, 3)
    mask = torch.zeros(2, 7, RNA_NUM_ATOMS, dtype=torch.bool)
    for backbone_index, atom_name in enumerate(
        ("P", "O5'", "C5'", "C4'", "C3'", "O3'")
    ):
        atom_index = RNA_ATOM_TO_INDEX[atom_name]
        coords[..., atom_index, :] = backbone[..., backbone_index, :]
        mask[..., atom_index] = True

    # Chi is unavailable in this backbone-only fixture; alpha..zeta still
    # round-trip exactly through the observed-coordinate target extractor.
    loss = torsion_parameter_loss(torsions, coords, mask)

    assert loss.item() < 1e-6


def test_direct_torsion_parameter_loss_is_periodic_and_has_finite_gradients():
    torch.manual_seed(62)
    target = torch.randn(1, 5, RNA_NUM_ATOMS, 3)
    mask = torch.ones(target.shape[:-1], dtype=torch.bool)
    input_ids = torch.tensor([[1, 2, 3, 4, 1]])
    predicted = torch.randn(1, 5, 7, 2, requires_grad=True)

    loss = torsion_parameter_loss(predicted, target, mask, input_ids)
    loss.backward()

    assert loss.item() >= 0.0
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad.abs().sum().item() > 0.0


def test_periodic_torsion_loss_supervises_cross_residue_epsilon_and_zeta():
    torch.manual_seed(107)
    target = torch.randn(1, 2, RNA_NUM_ATOMS, 3)
    mask = torch.zeros(target.shape[:-1], dtype=torch.bool)
    for residue, atoms in (
        (0, ("C4'", "C3'", "O3'")),
        (1, ("P", "O5'")),
    ):
        for atom in atoms:
            mask[0, residue, RNA_ATOM_TO_INDEX[atom]] = True
    pred = target.clone()
    matching = torsion_angle_loss(pred, target, mask)
    pred[0, 1, RNA_ATOM_TO_INDEX["P"]] += torch.tensor(
        [0.8, -0.4, 1.1]
    )
    changed = torsion_angle_loss(pred, target, mask)

    assert matching.item() < 1e-6
    assert changed.item() > matching.item() + 1e-4


def test_periodic_sugar_pucker_loss_matches_target_phase():
    torch.manual_seed(8)
    target = torch.randn(1, 4, RNA_NUM_ATOMS, 3)
    mask = torch.ones(target.shape[:-1], dtype=torch.bool)
    target_phase, valid = sugar_pucker_phase(target, mask)

    matching = sugar_pucker_phase_loss(target_phase, target, mask)
    opposite = sugar_pucker_phase_loss(-target_phase, target, mask)

    assert valid.all()
    assert matching.item() == pytest.approx(0.0, abs=1e-6)
    assert opposite.item() == pytest.approx(2.0, abs=1e-5)


def test_generated_c3_endo_sugar_phase_matches_head_convention():
    torch.manual_seed(63)
    config = RhoFoldConfig(
        d_model=16, pair_dim=8, msa_dim=8, nhead=4, pair_heads=2,
        num_e2e_layers=1, num_structure_layers=1, dim_feedforward=32,
        equivariant_layers=0, dropout=0.0,
    )
    model = RhoFoldModel(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        output = model(input_ids, return_aux=True)
    observed, valid = sugar_pucker_phase(
        output["coords"], chemical_atom_mask(input_ids)
    )
    predicted = output["sugar_pucker"]
    periodic_error = 1.0 - (observed * predicted).sum(dim=-1)

    assert valid.all()
    assert periodic_error[valid].mean().item() < 0.01


def test_coordinate_pucker_loss_compares_final_sugar_geometry():
    torch.manual_seed(64)
    target = torch.randn(1, 4, RNA_NUM_ATOMS, 3)
    mask = torch.ones(target.shape[:-1], dtype=torch.bool)
    pred = target.clone()
    matching = sugar_pucker_coordinate_loss(pred, target, mask)
    pred[..., RNA_ATOM_TO_INDEX["C2'"], 2] += 1.0
    perturbed = sugar_pucker_coordinate_loss(pred, target, mask)

    assert matching.item() < 1e-6
    assert perturbed.item() > matching.item()


def test_coordinate_pucker_loss_reaches_sugar_pucker_head():
    torch.manual_seed(65)
    config = RhoFoldConfig(
        d_model=16, pair_dim=8, msa_dim=8, nhead=4, pair_heads=2,
        num_e2e_layers=1, num_structure_layers=1, dim_feedforward=32,
        equivariant_layers=0, dropout=0.0,
    )
    input_ids = torch.tensor([[1, 2, 3, 4]])
    model = RhoFoldModel(config)
    output = model(input_ids, return_aux=True)
    target = output["coords"].detach().clone()
    target[..., RNA_ATOM_TO_INDEX["C2'"], 2] += 0.5
    mask = chemical_atom_mask(input_ids)

    loss = sugar_pucker_coordinate_loss(output["coords"], target, mask)
    loss.backward()
    gradient = model.structure_module.sugar_pucker_head.weight.grad

    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


def test_inter_residue_phosphodiester_bond_has_physical_target():
    coords = torch.zeros((1, 2, RNA_NUM_ATOMS, 3))
    mask = torch.zeros(coords.shape[:-1], dtype=torch.bool)
    o3 = RNA_ATOM_NAMES.index("O3'")
    p = RNA_ATOM_NAMES.index("P")
    coords[0, 0, o3] = torch.tensor([0.0, 0.0, 0.0])
    coords[0, 1, p] = torch.tensor([1.6, 0.0, 0.0])
    mask[0, 0, o3] = True
    mask[0, 1, p] = True

    assert inter_residue_geometry_loss(coords, mask).item() == pytest.approx(0.0)


def test_inter_residue_geometry_enforces_both_phosphodiester_angles():
    coords = torch.zeros((1, 2, RNA_NUM_ATOMS, 3))
    mask = torch.zeros(coords.shape[:-1], dtype=torch.bool)
    c3 = RNA_ATOM_TO_INDEX["C3'"]
    o3 = RNA_ATOM_TO_INDEX["O3'"]
    p = RNA_ATOM_TO_INDEX["P"]
    o5 = RNA_ATOM_TO_INDEX["O5'"]
    coords[0, 0, o3] = torch.tensor([0.0, 0.0, 0.0])
    coords[0, 0, c3] = 1.42 * torch.tensor(
        [math.cos(math.radians(120)), math.sin(math.radians(120)), 0.0]
    )
    coords[0, 1, p] = torch.tensor([1.60, 0.0, 0.0])
    coords[0, 1, o5] = coords[0, 1, p] + 1.60 * torch.tensor(
        [math.cos(math.radians(76)), math.sin(math.radians(76)), 0.0]
    )
    mask[0, 0, [c3, o3]] = True
    mask[0, 1, [p, o5]] = True
    baseline = inter_residue_geometry_loss(coords, mask)
    broken = coords.clone()
    broken[0, 1, o5] += torch.tensor([0.0, 0.0, 1.0])

    assert baseline.item() < 1e-5
    assert inter_residue_geometry_loss(broken, mask) > baseline + 1e-3


def test_all_atom_clash_excludes_covalent_o3_p_but_penalizes_other_adjacent_atoms():
    coords = torch.zeros((1, 2, RNA_NUM_ATOMS, 3))
    mask = torch.zeros(coords.shape[:-1], dtype=torch.bool)
    o3 = RNA_ATOM_NAMES.index("O3'")
    p = RNA_ATOM_NAMES.index("P")
    c4 = RNA_ATOM_NAMES.index("C4'")
    mask[0, 0, o3] = True
    mask[0, 1, p] = True

    covalent_only = steric_clash_loss(coords, mask)

    mask[0, :, c4] = True
    nonbonded_clash = steric_clash_loss(coords, mask)

    assert covalent_only.item() == pytest.approx(0.0)
    assert nonbonded_clash.item() > 0.0


def test_clash_loss_normalizes_per_atom_not_quadratic_pair_count():
    coords = torch.zeros((1, 4, RNA_NUM_ATOMS, 3))
    mask = torch.zeros(coords.shape[:-1], dtype=torch.bool)
    c4 = RNA_ATOM_TO_INDEX["C4'"]
    mask[0, :, c4] = True
    coords[0, 2, c4] = torch.tensor([10.0, 0.0, 0.0])
    coords[0, 3, c4] = torch.tensor([20.0, 0.0, 0.0])

    loss = steric_clash_loss(coords, mask)

    # Carbon-carbon threshold is 1.7 + 1.7 - 0.6 = 2.8 Å.  Only one
    # pair clashes, and its energy is normalized by four valid atoms rather
    # than all six possible pairs.
    assert loss.item() == pytest.approx(2.8 ** 2 / 4.0, rel=1e-6)


def test_single_clash_decays_linearly_not_quadratically_with_rna_length():
    c4 = RNA_ATOM_TO_INDEX["C4'"]
    losses = []
    for length in (2, 16):
        coords = torch.zeros((1, length, RNA_NUM_ATOMS, 3))
        mask = torch.zeros(coords.shape[:-1], dtype=torch.bool)
        mask[0, :, c4] = True
        coords[0, 2:, c4, 0] = (
            torch.arange(1, length - 1, dtype=coords.dtype) * 10.0
        )
        losses.append(steric_clash_loss(coords, mask))

    assert (losses[0] / losses[1]).item() == pytest.approx(8.0, rel=1e-6)


def test_clash_loss_weights_each_rna_equally_in_a_mixed_length_batch():
    coords = torch.zeros((2, 3, RNA_NUM_ATOMS, 3))
    mask = torch.zeros(coords.shape[:-1], dtype=torch.bool)
    c4 = RNA_ATOM_TO_INDEX["C4'"]
    mask[:, :2, c4] = True
    mask[1, 2, c4] = True
    coords[1, 2, c4] = torch.tensor([10.0, 0.0, 0.0])

    batch_loss = steric_clash_loss(coords, mask)
    first_loss = steric_clash_loss(coords[:1], mask[:1])
    second_loss = steric_clash_loss(coords[1:], mask[1:])

    assert batch_loss.item() == pytest.approx(
        ((first_loss + second_loss) / 2.0).item(), rel=1e-6
    )


def test_ccd_templates_and_glycosidic_chi_are_physically_resolved():
    model = RhoFoldModel(
        RhoFoldConfig(
            d_model=16,
            pair_dim=8,
            msa_dim=8,
            nhead=4,
            pair_heads=2,
            num_e2e_layers=1,
            num_structure_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            equivariant_layers=0,
        )
    ).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        output = model(input_ids, return_aux=True)
    coords = output["coords"]
    chemical_mask = chemical_atom_mask(input_ids)

    assert bond_length_loss(coords, chemical_mask, input_ids).item() < 1e-3
    for residue, base in enumerate(("A", "U", "C", "G")):
        glycosidic = "N9" if base in {"A", "G"} else "N1"
        reference = "C4" if base in {"A", "G"} else "C2"
        actual_chi = _dihedral(
            coords[:, residue, RNA_ATOM_TO_INDEX["O4'"]],
            coords[:, residue, RNA_ATOM_TO_INDEX["C1'"]],
            coords[:, residue, RNA_ATOM_TO_INDEX[glycosidic]],
            coords[:, residue, RNA_ATOM_TO_INDEX[reference]],
        )
        target_chi = torch.atan2(
            output["torsions"][:, residue, 6, 0],
            output["torsions"][:, residue, 6, 1],
        )
        periodic_error = 1.0 - torch.cos(actual_chi - target_chi)
        assert periodic_error.item() < 1e-6


def test_chemical_bond_graph_is_base_specific_and_symmetric():
    adjacency = chemical_bond_adjacency(torch.tensor([[1, 2, 3, 4]]))
    c1 = RNA_ATOM_TO_INDEX["C1'"]
    n1 = RNA_ATOM_TO_INDEX["N1"]
    n9 = RNA_ATOM_TO_INDEX["N9"]

    assert adjacency[0, 0, c1, n9] and adjacency[0, 0, n9, c1]
    assert not adjacency[0, 0, c1, n1]
    assert adjacency[0, 1, c1, n1] and not adjacency[0, 1, c1, n9]
    assert adjacency[0, 2, c1, n1] and not adjacency[0, 2, c1, n9]
    assert adjacency[0, 3, c1, n9] and not adjacency[0, 3, c1, n1]


def test_same_residue_nonbonded_clash_is_penalized_with_covalent_exclusions():
    model = RhoFoldModel(
        RhoFoldConfig(
            d_model=16, pair_dim=8, msa_dim=8, nhead=4, pair_heads=2,
            num_e2e_layers=1, num_structure_layers=1, dim_feedforward=32,
            dropout=0.0, equivariant_layers=0,
        )
    ).eval()
    input_ids = torch.tensor([[1]])
    with torch.no_grad():
        coords = model(input_ids)
    chemical_mask = chemical_atom_mask(input_ids)
    baseline = steric_clash_loss(coords, chemical_mask, input_ids)
    collided = coords.clone()
    collided[0, 0, RNA_ATOM_TO_INDEX["N6"]] = collided[
        0, 0, RNA_ATOM_TO_INDEX["O2'"]
    ]

    collision = steric_clash_loss(collided, chemical_mask, input_ids)

    assert baseline.item() < 0.05
    assert collision.item() > baseline.item() + 0.04


def test_chemical_clash_loss_is_residue_chunk_invariant():
    torch.manual_seed(10)
    input_ids = torch.tensor([[1, 2, 3, 4, 1]])
    coords = torch.randn(1, 5, RNA_NUM_ATOMS, 3)
    chemical_mask = chemical_atom_mask(input_ids)

    single = steric_clash_loss(
        coords, chemical_mask, input_ids, residue_chunk_size=1
    )
    chunked = steric_clash_loss(
        coords, chemical_mask, input_ids, residue_chunk_size=3
    )

    assert torch.allclose(single, chunked, atol=1e-6)


def test_padding_does_not_change_valid_residue_outputs_or_pair_features():
    torch.manual_seed(11)
    config = RhoFoldConfig(
        d_model=32,
        pair_dim=16,
        msa_dim=16,
        nhead=4,
        pair_heads=4,
        num_e2e_layers=1,
        num_structure_layers=1,
        dim_feedforward=64,
        recycle_iters=2,
        random_recycle_training=False,
        dropout=0.0,
    )
    model = RhoFoldModel(config).eval()
    short_ids = torch.tensor([[1, 2, 3, 4]])
    padded_ids = torch.tensor([[1, 2, 3, 4, 0, 0]])
    with torch.inference_mode():
        short = model(short_ids, return_aux=True)
        padded = model(padded_ids, return_aux=True)

    assert torch.allclose(short["coords"], padded["coords"][:, :4], atol=1e-5)
    assert torch.allclose(short["pair_embedding"], padded["pair_embedding"][:, :4, :4], atol=1e-5)
    assert torch.count_nonzero(padded["pair_embedding"][:, 4:]) == 0


def test_activation_checkpointing_preserves_outputs_and_gradients():
    torch.manual_seed(111)
    base_config = RhoFoldConfig(
        d_model=16,
        pair_dim=8,
        msa_dim=8,
        nhead=4,
        pair_heads=2,
        num_e2e_layers=2,
        num_structure_layers=1,
        dim_feedforward=32,
        recycle_iters=1,
        random_recycle_training=False,
        dropout=0.1,
        activation_checkpointing=False,
    )
    direct = RhoFoldModel(base_config).train()
    checkpointed = RhoFoldModel(
        RhoFoldConfig(
            **{
                **base_config.__dict__,
                "activation_checkpointing": True,
            }
        )
    ).train()
    checkpointed.load_state_dict(direct.state_dict())
    input_ids = torch.tensor([[1, 2, 3, 4, 1]])

    torch.manual_seed(112)
    direct_output = direct(input_ids, return_aux=True)
    torch.manual_seed(112)
    checkpointed_output = checkpointed(input_ids, return_aux=True)
    direct_loss = (
        direct_output["coords"].square().mean()
        + direct_output["pair_embedding"].square().mean()
    )
    checkpointed_loss = (
        checkpointed_output["coords"].square().mean()
        + checkpointed_output["pair_embedding"].square().mean()
    )
    direct_loss.backward()
    checkpointed_loss.backward()

    assert torch.allclose(
        direct_output["coords"], checkpointed_output["coords"], atol=1e-6
    )
    assert torch.allclose(
        direct_output["pair_embedding"],
        checkpointed_output["pair_embedding"],
        atol=1e-6,
    )
    for direct_parameter, checkpointed_parameter in zip(
        direct.parameters(), checkpointed.parameters()
    ):
        if direct_parameter.grad is None:
            assert checkpointed_parameter.grad is None
        else:
            assert checkpointed_parameter.grad is not None
            assert torch.allclose(
                direct_parameter.grad,
                checkpointed_parameter.grad,
                atol=1e-5,
                rtol=1e-4,
            )


def test_directed_pair_channels_and_symmetric_distance_head_are_separate():
    config = RhoFoldConfig(
        d_model=16,
        pair_dim=8,
        msa_dim=8,
        nhead=4,
        pair_heads=2,
        num_e2e_layers=1,
        num_structure_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    model = RhoFoldModel(config).eval()
    with torch.inference_mode():
        output = model(torch.tensor([[1, 2, 3, 4]]), return_aux=True)

    directed = output["pair_embedding"]
    distances = output["pair_distance_logits"]
    assert not torch.allclose(directed, directed.transpose(1, 2))
    assert torch.allclose(distances, distances.transpose(1, 2), atol=1e-6)


def test_sequence_attention_uses_pair_bias_and_masks_padding_edges():
    torch.manual_seed(101)
    config = RhoFoldConfig(
        d_model=8, pair_dim=4, msa_dim=8, nhead=2, pair_heads=2,
        num_e2e_layers=1, num_structure_layers=1,
        dim_feedforward=16, dropout=0.0,
    )
    attention = PairBiasedSelfAttention(config).eval()
    seq = torch.randn(1, 3, 8)
    pair = torch.zeros(1, 3, 3, 4, requires_grad=True)
    pair_mask = torch.tensor(
        [[[True, True, False], [True, True, False], [False, False, False]]]
    )
    baseline = attention(seq, pair, pair_mask)
    changed_pair = pair.detach().clone()
    changed_pair[0, 0, 1] = torch.tensor([4.0, -3.0, 2.0, 1.0])
    changed = attention(seq, changed_pair, pair_mask)
    padded_pair = changed_pair.clone()
    padded_pair[0, :, 2] = 1e4
    padded_pair[0, 2, :] = -1e4
    padded = attention(seq, padded_pair, pair_mask)

    assert not torch.allclose(baseline[:, :2], changed[:, :2])
    assert torch.allclose(changed[:, :2], padded[:, :2], atol=1e-6)
    baseline[:, :2].square().sum().backward()
    assert pair.grad is not None
    assert pair.grad[pair_mask].abs().sum() > 0
    assert pair.grad[~pair_mask].abs().sum() == 0


def test_pair_attention_pooling_ignores_padding_and_zeroes_empty_rows():
    torch.manual_seed(102)
    config = RhoFoldConfig(
        d_model=8, pair_dim=4, msa_dim=8, nhead=2, pair_heads=2,
        num_e2e_layers=1, num_structure_layers=1,
        dim_feedforward=16, dropout=0.0,
    )
    pooling = PairAttentionPooling(config)
    pair = torch.randn(1, 3, 3, 4)
    pair_mask = torch.tensor(
        [[[True, True, False], [True, True, False], [False, False, False]]]
    )
    baseline = pooling(pair, pair_mask)
    polluted = pair.clone()
    polluted[:, :, 2] = 1e6
    polluted[:, 2, :] = -1e6
    changed = pooling(polluted, pair_mask)

    assert torch.allclose(baseline[:, :2], changed[:, :2], atol=1e-6)
    assert torch.equal(changed[:, 2], torch.zeros_like(changed[:, 2]))


def test_pair_transition_masks_padding_and_updates_valid_pairs():
    torch.manual_seed(103)
    config = RhoFoldConfig(
        d_model=8, pair_dim=4, msa_dim=8, nhead=2, pair_heads=2,
        num_e2e_layers=1, num_structure_layers=1,
        dim_feedforward=16, dropout=0.0,
    )
    transition = PairTransition(config)
    pair = torch.randn(1, 3, 3, 4, requires_grad=True)
    pair_mask = torch.tensor(
        [[[True, True, False], [True, True, False], [False, False, False]]]
    )
    update = transition(pair, pair_mask)
    update.sum().backward()

    assert update[pair_mask].abs().sum() > 0
    assert torch.equal(
        update[~pair_mask], torch.zeros_like(update[~pair_mask])
    )
    assert pair.grad is not None
    assert pair.grad[~pair_mask].abs().sum() == 0


def test_outer_product_update_represents_cross_channel_products():
    config = RhoFoldConfig(
        d_model=8,
        pair_dim=4,
        msa_dim=8,
        nhead=2,
        pair_heads=2,
        triangle_hidden_dim=8,
        triangle_chunk_size=1,
    )
    update = OuterProductMean(config)
    with torch.no_grad():
        for parameter in update.parameters():
            parameter.zero_()
        update.left.weight[0, 0] = 1.0
        update.right.weight[1, 1] = 1.0
        update.output.weight[0, 1] = 1.0  # flattened left-0 × right-1
    seq = torch.zeros(1, 2, config.d_model)
    seq[0, 0, 0] = 2.0
    seq[0, 1, 1] = 3.0
    pair_mask = torch.ones(1, 2, 2, dtype=torch.bool)

    pair = update(seq, pair_mask)

    assert pair[0, 0, 1, 0].item() == pytest.approx(6.0)
    assert torch.count_nonzero(pair[..., 1:]) == 0


def test_outer_product_chunking_preserves_values_and_input_gradients():
    torch.manual_seed(14)
    config_fine = RhoFoldConfig(
        d_model=8,
        pair_dim=4,
        msa_dim=8,
        nhead=2,
        pair_heads=2,
        triangle_hidden_dim=8,
        triangle_chunk_size=1,
    )
    config_coarse = RhoFoldConfig(
        **{
            **config_fine.__dict__,
            "triangle_chunk_size": 3,
        }
    )
    fine = OuterProductMean(config_fine)
    coarse = OuterProductMean(config_coarse)
    coarse.load_state_dict(fine.state_dict())
    seq_fine = torch.randn(2, 5, 8, requires_grad=True)
    seq_coarse = seq_fine.detach().clone().requires_grad_(True)
    residue_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )
    pair_mask = residue_mask.unsqueeze(2) & residue_mask.unsqueeze(1)

    output_fine = fine(seq_fine, pair_mask)
    output_coarse = coarse(seq_coarse, pair_mask)
    output_fine.square().sum().backward()
    output_coarse.square().sum().backward()

    assert torch.allclose(output_fine, output_coarse, atol=1e-6)
    assert torch.allclose(seq_fine.grad, seq_coarse.grad, atol=1e-6)


@pytest.mark.parametrize(
    ("outgoing", "expected_edges", "reverse_edges"),
    [
        (True, ((0, 2), (1, 2)), ((2, 0), (2, 1))),
        (False, ((2, 0), (2, 1)), ((0, 2), (1, 2))),
    ],
)
def test_triangle_multiplication_uses_directed_third_edges(
    outgoing,
    expected_edges,
    reverse_edges,
):
    torch.manual_seed(120)
    config = RhoFoldConfig(
        d_model=16,
        pair_dim=8,
        msa_dim=8,
        nhead=4,
        pair_heads=2,
        triangle_hidden_dim=8,
        triangle_chunk_size=2,
    )
    update = TriangleMultiplicativeUpdate(
        config, outgoing=outgoing
    )
    pair = torch.randn(1, 3, 3, 8, requires_grad=True)
    pair_mask = torch.ones(1, 3, 3, dtype=torch.bool)

    update(pair, pair_mask)[0, 0, 1].square().sum().backward()

    for left, right in expected_edges:
        assert pair.grad[0, left, right].abs().sum() > 0
    for left, right in reverse_edges:
        assert pair.grad[0, left, right].abs().sum() == 0


@pytest.mark.parametrize("outgoing", [True, False])
def test_triangle_multiplication_masks_padding_third_edges(outgoing):
    torch.manual_seed(121)
    config = RhoFoldConfig(
        d_model=16,
        pair_dim=8,
        msa_dim=8,
        nhead=4,
        pair_heads=2,
        triangle_hidden_dim=8,
    )
    update = TriangleMultiplicativeUpdate(
        config, outgoing=outgoing
    )
    pair = torch.randn(1, 3, 3, 8, requires_grad=True)
    residue_mask = torch.tensor([[True, True, False]])
    pair_mask = residue_mask.unsqueeze(2) & residue_mask.unsqueeze(1)

    update(pair, pair_mask)[0, 0, 1].square().sum().backward()

    assert pair.grad[0, 2].abs().sum() == 0
    assert pair.grad[0, :, 2].abs().sum() == 0


@pytest.mark.parametrize("starting", [True, False])
def test_triangle_attention_uses_the_third_pair_edge_as_bias(starting):
    torch.manual_seed(12)
    config = RhoFoldConfig(
        d_model=16,
        pair_dim=8,
        msa_dim=8,
        nhead=4,
        pair_heads=2,
        num_e2e_layers=1,
        num_structure_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        triangle_chunk_size=2,
    )
    attention = TriangleAttention(config, starting=starting).eval()
    pair = torch.randn(1, 4, 4, config.pair_dim)
    pair_mask = torch.ones(1, 4, 4, dtype=torch.bool)
    perturbed = pair.clone()
    perturbed[:, 2, 3, 0] += 3.0

    original = attention(pair, pair_mask)
    changed = attention(perturbed, pair_mask)
    unaffected_axis = (
        (original[:, 0], changed[:, 0])
        if starting
        else (original[:, :, 0], changed[:, :, 0])
    )

    assert not torch.allclose(*unaffected_axis)


@pytest.mark.parametrize("starting", [True, False])
def test_triangle_attention_chunking_preserves_values_and_input_gradients(
    starting,
):
    torch.manual_seed(13)
    base_config = dict(
        d_model=16,
        pair_dim=8,
        msa_dim=8,
        nhead=4,
        pair_heads=2,
        num_e2e_layers=1,
        num_structure_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    fine = TriangleAttention(
        RhoFoldConfig(**base_config, triangle_chunk_size=1),
        starting=starting,
    )
    coarse = TriangleAttention(
        RhoFoldConfig(**base_config, triangle_chunk_size=3),
        starting=starting,
    )
    coarse.load_state_dict(fine.state_dict())
    pair_fine = torch.randn(2, 4, 4, 8, requires_grad=True)
    pair_coarse = pair_fine.detach().clone().requires_grad_(True)
    residue_mask = torch.tensor(
        [[True, True, True, True], [True, True, True, False]]
    )
    pair_mask = residue_mask.unsqueeze(2) & residue_mask.unsqueeze(1)

    output_fine = fine(pair_fine, pair_mask)
    output_coarse = coarse(pair_coarse, pair_mask)
    output_fine.square().sum().backward()
    output_coarse.square().sum().backward()

    assert torch.allclose(output_fine, output_coarse, atol=1e-6)
    assert torch.allclose(pair_fine.grad, pair_coarse.grad, atol=1e-6)


def test_frame_structure_head_avoids_collapsed_adjacent_residues():
    config = RhoFoldConfig(
        d_model=16,
        pair_dim=8,
        msa_dim=8,
        nhead=4,
        pair_heads=2,
        num_e2e_layers=1,
        num_structure_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    model = RhoFoldModel(config).eval()
    with torch.inference_mode():
        coords = model(torch.tensor([[1, 2, 3, 4, 1]]))
    c1 = coords[0, :, RNA_ATOM_NAMES.index("C1'")]
    adjacent = torch.linalg.norm(c1[1:] - c1[:-1], dim=-1)

    assert torch.all(adjacent > 4.5)
    assert torch.all(adjacent < 7.0)


def test_internal_coordinate_backbone_preserves_covalent_bond_lengths():
    torsions = torch.zeros(1, 4, 7, 2)
    torsions[..., 1] = 1.0
    padding_mask = torch.zeros(1, 4, dtype=torch.bool)
    rotation = torch.eye(3).unsqueeze(0)

    backbone, frames, _ = build_rna_backbone(torsions, padding_mask, rotation)
    p, o5, c5, c4, c3, o3 = backbone.unbind(dim=-2)

    assert torch.allclose(torch.linalg.norm(o5 - p, dim=-1), torch.full((1, 4), 1.60), atol=1e-5)
    assert torch.allclose(torch.linalg.norm(c5 - o5, dim=-1), torch.full((1, 4), 1.43), atol=1e-5)
    assert torch.allclose(torch.linalg.norm(c4 - c5, dim=-1), torch.full((1, 4), 1.52), atol=1e-5)
    assert torch.allclose(torch.linalg.norm(c3 - c4, dim=-1), torch.full((1, 4), 1.53), atol=1e-5)
    assert torch.allclose(torch.linalg.norm(o3 - c3, dim=-1), torch.full((1, 4), 1.42), atol=1e-5)
    assert torch.allclose(
        torch.linalg.norm(p[:, 1:] - o3[:, :-1], dim=-1),
        torch.full((1, 3), 1.60),
        atol=1e-5,
    )
    identity = torch.eye(3).view(1, 1, 3, 3)
    assert torch.allclose(
        torch.matmul(frames.transpose(-2, -1), frames),
        identity.expand_as(frames),
        atol=1e-5,
    )


def test_structure_coordinates_backpropagate_to_torsion_pucker_and_base_orientation_heads():
    torch.manual_seed(17)
    config = RhoFoldConfig(
        d_model=16, pair_dim=8, msa_dim=8, nhead=4, pair_heads=2,
        num_e2e_layers=1, num_structure_layers=1, dim_feedforward=32,
        equivariant_layers=0, dropout=0.0,
    )
    model = RhoFoldModel(config)
    output = model(torch.tensor([[1, 2, 3, 4]]), return_aux=True)

    assert output["sugar_pucker"].shape == (1, 4, 2)
    assert output["base_orientation"].shape == (1, 4, 3, 3)
    output["coords"].square().mean().backward()

    structure = model.structure_module
    assert structure.torsion_head.weight.grad is not None
    assert structure.torsion_head.weight.grad.abs().sum() > 0
    assert structure.sugar_pucker_head.weight.grad is not None
    assert structure.sugar_pucker_head.weight.grad.abs().sum() > 0
    assert structure.base_orientation_head.weight.grad is not None
    assert structure.base_orientation_head.weight.grad.abs().sum() > 0


def test_structure_module_coordinates_receive_full_pair_tensor_gradients():
    torch.manual_seed(104)
    config = RhoFoldConfig(
        d_model=16,
        pair_dim=8,
        msa_dim=8,
        nhead=4,
        pair_heads=2,
        num_e2e_layers=1,
        num_structure_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        equivariant_layers=1,
    )
    model = RhoFoldModel(config)
    seq = torch.randn(1, 4, config.d_model)
    pair = torch.randn(
        1, 4, 4, config.pair_dim, requires_grad=True
    )
    input_ids = torch.tensor([[1, 2, 3, 4]])
    padding_mask = torch.zeros(1, 4, dtype=torch.bool)
    pair_mask = torch.ones(1, 4, 4, dtype=torch.bool)
    base_probabilities = torch.nn.functional.one_hot(
        input_ids - 1, num_classes=4
    ).float()

    output = model.structure_module(
        seq,
        pair,
        input_ids,
        padding_mask,
        pair_mask,
        base_probabilities,
    )
    output["coords"].square().mean().backward()

    assert pair.grad is not None
    off_diagonal = ~torch.eye(4, dtype=torch.bool).unsqueeze(0)
    assert pair.grad[off_diagonal].abs().sum() > 0


def test_internal_coordinate_sugar_ring_is_closed_with_physical_bonds():
    config = RhoFoldConfig(
        d_model=16, pair_dim=8, msa_dim=8, nhead=4, pair_heads=2,
        num_e2e_layers=1, num_structure_layers=1, dim_feedforward=32,
        equivariant_layers=0, dropout=0.0,
    )
    model = RhoFoldModel(config).eval()
    with torch.inference_mode():
        coords = model(torch.tensor([[1, 2, 3, 4]]))
    ring = ("C4'", "C3'", "C2'", "C1'", "O4'", "C4'")
    lengths = []
    for left, right in zip(ring, ring[1:]):
        lengths.append(torch.linalg.norm(
            coords[..., RNA_ATOM_NAMES.index(left), :]
            - coords[..., RNA_ATOM_NAMES.index(right), :],
            dim=-1,
        ))
    lengths = torch.stack(lengths)

    assert torch.all(lengths > 1.40)
    assert torch.all(lengths < 1.66)


def test_internal_coordinate_template_has_physical_glycosidic_angles():
    config = RhoFoldConfig(
        d_model=16, pair_dim=8, msa_dim=8, nhead=4, pair_heads=2,
        num_e2e_layers=1, num_structure_layers=1, dim_feedforward=32,
        equivariant_layers=0, dropout=0.0,
    )
    model = RhoFoldModel(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        coords = model(input_ids)
    c1 = RNA_ATOM_TO_INDEX["C1'"]
    angles = []
    for residue, base in enumerate(("A", "U", "C", "G")):
        glycosidic = "N9" if base in ("A", "G") else "N1"
        center = coords[0, residue, c1]
        for sugar_atom in ("O4'", "C2'"):
            left = coords[0, residue, RNA_ATOM_TO_INDEX[sugar_atom]] - center
            right = (
                coords[0, residue, RNA_ATOM_TO_INDEX[glycosidic]]
                - center
            )
            angles.append(
                torch.rad2deg(
                    torch.acos(
                        torch.nn.functional.cosine_similarity(
                            left.unsqueeze(0), right.unsqueeze(0)
                        )
                    )
                )
            )

    angles = torch.stack(angles)
    assert torch.all(angles > 103.0)
    assert torch.all(angles < 118.0)
    physical_mask = chemical_atom_mask(input_ids)
    assert bond_angle_loss(
        coords, physical_mask, input_ids
    ).item() < 0.1


def test_amp_keeps_internal_coordinate_recurrence_in_float32():
    config = RhoFoldConfig(
        d_model=16, pair_dim=8, msa_dim=8, nhead=4, pair_heads=2,
        num_e2e_layers=1, num_structure_layers=1, dim_feedforward=32,
        equivariant_layers=0, dropout=0.0,
    )
    model = RhoFoldModel(config).eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cpu", dtype=torch.bfloat16
    ):
        coords = model(torch.tensor([[1, 2, 3, 4, 1, 2]]))
    o3 = RNA_ATOM_NAMES.index("O3'")
    p = RNA_ATOM_NAMES.index("P")
    inter_residue_bonds = torch.linalg.norm(
        coords[:, :-1, o3] - coords[:, 1:, p], dim=-1
    )

    assert coords.dtype == torch.float32
    assert torch.isfinite(coords).all()
    assert torch.allclose(
        inter_residue_bonds,
        torch.full_like(inter_residue_bonds, 1.60),
        atol=1e-4,
    )


def test_internal_coordinate_refinement_is_se3_invariant_and_preserves_norm():
    torch.manual_seed(191)
    config = RhoFoldConfig(
        d_model=16, pair_dim=8, msa_dim=8, nhead=4, pair_heads=2,
        num_e2e_layers=1, num_structure_layers=1, dim_feedforward=32,
    )
    layer = EquivariantInternalCoordinateRefinement(config).eval()
    with torch.no_grad():
        layer.torsion_delta.weight.normal_(std=0.1)
    torsions = torch.randn(1, 5, 7, 2)
    torsions = torch.nn.functional.normalize(torsions, dim=-1)
    origins = torch.randn(1, 5, 3)
    pair = torch.randn(1, 5, 5, 8)
    pair_mask = torch.ones(1, 5, 5, dtype=torch.bool)
    rotation = torch.linalg.qr(torch.randn(3, 3)).Q
    if torch.det(rotation) < 0:
        rotation[:, -1] *= -1
    translation = torch.tensor([2.0, -3.0, 1.0])
    transformed_origins = origins @ rotation.T + translation

    expected = layer(torsions, origins, pair, pair_mask)
    actual = layer(torsions, transformed_origins, pair, pair_mask)

    assert torch.allclose(actual, expected, atol=1e-5)
    assert torch.allclose(
        torch.linalg.norm(actual, dim=-1),
        torch.ones_like(actual[..., 0]),
        atol=1e-5,
    )


def test_full_all_atom_structure_is_equivariant_to_initial_global_rotation():
    torch.manual_seed(106)
    config = RhoFoldConfig(
        d_model=16,
        pair_dim=8,
        msa_dim=8,
        nhead=4,
        pair_heads=2,
        num_e2e_layers=1,
        num_structure_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        equivariant_layers=0,
    )
    model = RhoFoldModel(config)
    structure = model.structure_module
    hidden = torch.randn(1, 4, config.d_model)
    padding_mask = torch.zeros(1, 4, dtype=torch.bool)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    base_probabilities = torch.nn.functional.one_hot(
        input_ids - 1, num_classes=4
    ).float()
    with torch.no_grad():
        structure.rotation_head.weight.zero_()
        structure.rotation_head.bias.copy_(
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        )
        baseline = structure._predict_geometry(
            hidden, padding_mask, base_probabilities
        )
        quarter_turn = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        structure.rotation_head.bias.copy_(
            torch.cat((quarter_turn[:, 0], quarter_turn[:, 1]))
        )
        rotated = structure._predict_geometry(
            hidden, padding_mask, base_probabilities
        )

    expected_coords = torch.einsum(
        "ij,blaj->blai", quarter_turn, baseline["coords"]
    )
    expected_origins = torch.einsum(
        "ij,blj->bli", quarter_turn, baseline["origins"]
    )
    expected_frames = torch.einsum(
        "ij,bljk->blik", quarter_turn, baseline["frames"]
    )
    assert torch.allclose(rotated["coords"], expected_coords, atol=2e-4)
    assert torch.allclose(
        rotated["origins"], expected_origins, atol=2e-4
    )
    assert torch.allclose(
        rotated["frames"], expected_frames, atol=2e-4
    )


def test_default_equivariant_refinement_preserves_backbone_covalent_geometry():
    torch.manual_seed(192)
    config = RhoFoldConfig(
        d_model=16, pair_dim=8, msa_dim=8, nhead=4, pair_heads=2,
        num_e2e_layers=1, num_structure_layers=1, dim_feedforward=32,
        dropout=0.0, equivariant_layers=2,
    )
    model = RhoFoldModel(config).eval()
    with torch.inference_mode():
        coords = model(torch.tensor([[1, 2, 3, 4, 1, 2]]))
    o3 = RNA_ATOM_TO_INDEX["O3'"]
    p = RNA_ATOM_TO_INDEX["P"]
    inter_residue_bonds = torch.linalg.norm(
        coords[:, :-1, o3] - coords[:, 1:, p], dim=-1
    )

    assert torch.allclose(
        inter_residue_bonds,
        torch.full_like(inter_residue_bonds, 1.60),
        atol=1e-4,
    )


def test_internal_coordinate_refiners_receive_coordinate_gradients():
    torch.manual_seed(193)
    config = RhoFoldConfig(
        d_model=16, pair_dim=8, msa_dim=8, nhead=4, pair_heads=2,
        num_e2e_layers=1, num_structure_layers=1, dim_feedforward=32,
        dropout=0.0, equivariant_layers=2,
    )
    model = RhoFoldModel(config)

    coords = model(torch.tensor([[1, 2, 3, 4, 1]]))
    coords.square().mean().backward()

    for refiner in model.structure_module.refiners:
        gradient = refiner.torsion_delta.weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum().item() > 0.0


def test_recycling_features_are_invariant_to_global_rigid_transform():
    torch.manual_seed(20)
    config = RhoFoldConfig(
        d_model=16, pair_dim=8, msa_dim=8, nhead=4, pair_heads=2,
        num_e2e_layers=1, num_structure_layers=1, dim_feedforward=32,
    )
    recycling = RecyclingEmbedder(config).eval()
    coords = torch.randn(2, 5, RNA_NUM_ATOMS, 3)
    pair_mask = torch.ones(2, 5, 5, dtype=torch.bool)
    atom_mask = torch.ones(coords.shape[:-1], dtype=torch.bool)
    transformed = apply_random_rigid_augmentation(coords, atom_mask)

    previous_seq = torch.randn(2, 5, config.d_model)
    previous_pair = torch.randn(2, 5, 5, config.pair_dim)
    seq, pair = recycling(
        previous_seq, previous_pair, coords, pair_mask
    )
    transformed_seq, transformed_pair = recycling(
        previous_seq, previous_pair, transformed, pair_mask
    )

    assert torch.allclose(seq, transformed_seq, atol=1e-5)
    assert torch.allclose(pair, transformed_pair, atol=1e-5)


def test_recycling_preserves_previous_sequence_and_pair_information():
    torch.manual_seed(201)
    config = RhoFoldConfig(
        d_model=16, pair_dim=8, msa_dim=8, nhead=4, pair_heads=2,
        num_e2e_layers=1, num_structure_layers=1, dim_feedforward=32,
    )
    recycling = RecyclingEmbedder(config).eval()
    coords = torch.randn(1, 5, RNA_NUM_ATOMS, 3)
    pair_mask = torch.ones(1, 5, 5, dtype=torch.bool)
    seq = torch.randn(1, 5, config.d_model)
    pair = torch.randn(1, 5, 5, config.pair_dim)

    baseline_seq, baseline_pair = recycling(
        seq, pair, coords, pair_mask
    )
    changed_seq, _ = recycling(
        seq + 0.5 * torch.randn_like(seq), pair, coords, pair_mask
    )
    _, changed_pair = recycling(
        seq, pair + 0.5 * torch.randn_like(pair), coords, pair_mask
    )

    assert not torch.allclose(baseline_seq, changed_seq)
    assert not torch.allclose(baseline_pair, changed_pair)


def test_recycling_masks_previous_padding_features():
    torch.manual_seed(202)
    config = RhoFoldConfig(
        d_model=16, pair_dim=8, msa_dim=8, nhead=4, pair_heads=2,
        num_e2e_layers=1, num_structure_layers=1, dim_feedforward=32,
    )
    recycling = RecyclingEmbedder(config).eval()
    seq = torch.randn(1, 5, config.d_model)
    pair = torch.randn(1, 5, 5, config.pair_dim)
    coords = torch.randn(1, 5, RNA_NUM_ATOMS, 3)
    residue_mask = torch.tensor([[True, True, True, False, False]])
    pair_mask = residue_mask.unsqueeze(2) & residue_mask.unsqueeze(1)
    changed_seq = seq.clone()
    changed_pair = pair.clone()
    changed_seq[:, 3:] = 1e6
    changed_pair[:, 3:] = -1e6
    changed_pair[:, :, 3:] = 1e6

    baseline = recycling(seq, pair, coords, pair_mask)
    changed = recycling(
        changed_seq, changed_pair, coords, pair_mask
    )

    assert torch.allclose(
        baseline[0][:, :3], changed[0][:, :3], atol=1e-6
    )
    assert torch.allclose(
        baseline[1][:, :3, :3], changed[1][:, :3, :3], atol=1e-6
    )
    assert torch.count_nonzero(changed[0][:, 3:]) == 0
    assert torch.count_nonzero(changed[1][:, 3:]) == 0
    assert torch.count_nonzero(changed[1][:, :, 3:]) == 0


def test_joint_recycling_features_receive_finite_training_gradients():
    torch.manual_seed(203)
    config = RhoFoldConfig(
        d_model=16,
        pair_dim=8,
        msa_dim=8,
        nhead=4,
        pair_heads=2,
        num_e2e_layers=1,
        num_structure_layers=1,
        dim_feedforward=32,
        recycle_iters=2,
        recycle_stop_gradient=True,
        random_recycle_training=False,
        dropout=0.0,
        equivariant_layers=0,
    )
    model = RhoFoldModel(config).train()
    output = model(
        torch.tensor([[1, 2, 3, 4, 1]]),
        return_aux=True,
        recycle_iters=2,
    )
    loss = (
        output["coords"].square().mean()
        + output["sequence_embedding"].square().mean()
        + output["pair_embedding"].square().mean()
    )
    loss.backward()

    parameters = dict(model.recycling.named_parameters())
    for name in (
        "seq_scale",
        "pair_scale",
        "seq_norm.weight",
        "pair_norm.weight",
        "dist_to_pair.0.weight",
        "pair_to_seq.weight",
    ):
        gradient = parameters[name].grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum().item() > 0.0


def test_invariant_point_attention_is_invariant_to_global_rigid_transform():
    torch.manual_seed(21)
    config = RhoFoldConfig(
        d_model=16, pair_dim=8, msa_dim=8, nhead=4, pair_heads=2,
        num_e2e_layers=1, num_structure_layers=1, dim_feedforward=32,
        dropout=0.0,
    )
    ipa = InvariantPointAttention(config).eval()
    seq = torch.randn(2, 5, 16)
    pair = torch.randn(2, 5, 5, 8)
    pair_mask = torch.ones(2, 5, 5, dtype=torch.bool)
    frame_seed = torch.randn(2, 5, 6)
    rotations = rotation_6d_to_matrix(frame_seed)
    origins = torch.randn(2, 5, 3)
    global_rotation = rotation_6d_to_matrix(torch.randn(2, 6))
    translation = torch.randn(2, 3)
    transformed_rotations = torch.einsum(
        "bij,bljk->blik", global_rotation, rotations
    )
    transformed_origins = (
        torch.einsum("bij,blj->bli", global_rotation, origins)
        + translation.unsqueeze(1)
    )

    original = ipa(seq, pair, rotations, origins, pair_mask)
    transformed = ipa(
        seq, pair, transformed_rotations, transformed_origins, pair_mask
    )

    assert torch.allclose(original, transformed, atol=1e-5)


def test_recycle_counts_are_finite_and_keep_output_contract():
    config = RhoFoldConfig(
        d_model=16,
        pair_dim=8,
        msa_dim=8,
        nhead=4,
        pair_heads=2,
        num_e2e_layers=1,
        num_structure_layers=1,
        dim_feedforward=32,
        recycle_iters=3,
        random_recycle_training=False,
        dropout=0.0,
    )
    model = RhoFoldModel(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 1]])
    with torch.inference_mode():
        outputs = [
            model(input_ids, return_aux=True, recycle_iters=count)
            for count in (1, 2, 3)
        ]

    assert all(output["coords"].shape == (1, 5, RNA_NUM_ATOMS, 3) for output in outputs)
    assert all(torch.isfinite(output["coords"]).all() for output in outputs)
    assert not torch.allclose(outputs[0]["coords"], outputs[-1]["coords"])


def test_training_randomly_samples_recycle_counts_but_evaluation_uses_maximum():
    config = RhoFoldConfig(
        d_model=16,
        pair_dim=8,
        msa_dim=8,
        nhead=4,
        pair_heads=2,
        num_e2e_layers=1,
        num_structure_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        recycle_iters=3,
        random_recycle_training=True,
    )
    model = RhoFoldModel(config).train()
    torch.manual_seed(105)
    sampled = {
        model._resolve_recycle_iterations(None) for _ in range(100)
    }

    assert sampled == {1, 2, 3}
    model.eval()
    assert model._resolve_recycle_iterations(None) == 3


@pytest.mark.parametrize(
    ("stop_gradient", "expected_grad_modes"),
    [
        (True, [False, False, True]),
        (False, [True, True, True]),
    ],
)
def test_recycle_stop_gradient_disables_whole_nonfinal_iteration_graphs(
    stop_gradient,
    expected_grad_modes,
):
    config = RhoFoldConfig(
        d_model=16,
        pair_dim=8,
        msa_dim=8,
        nhead=4,
        pair_heads=2,
        num_e2e_layers=1,
        num_structure_layers=1,
        dim_feedforward=32,
        recycle_iters=3,
        recycle_stop_gradient=stop_gradient,
        random_recycle_training=False,
        dropout=0.0,
        equivariant_layers=0,
    )
    model = RhoFoldModel(config).train()
    grad_modes = []
    handle = model.e2eformer[0].register_forward_pre_hook(
        lambda module, inputs: grad_modes.append(torch.is_grad_enabled())
    )

    output = model(
        torch.tensor([[1, 2, 3, 4]]),
        recycle_iters=3,
    )
    output.square().mean().backward()
    handle.remove()

    assert grad_modes == expected_grad_modes
    assert model.structure_module.torsion_head.weight.grad is not None


def test_single_c1_label_losses_train_the_c1_output_without_shape_errors():
    torch.manual_seed(23)
    pred = torch.randn(1, 5, RNA_NUM_ATOMS, 3, requires_grad=True)
    target = torch.randn(1, 5, 3)
    mask = torch.ones(1, 5, dtype=torch.bool)
    logits = torch.zeros(1, 5, 5, 8, requires_grad=True)
    loss = (
        masked_coordinate_mse(pred, target, mask)
        + kabsch_aligned_coordinate_loss(pred, target, mask)
        + masked_pairwise_distance_mse(pred, target, mask)
        + pair_distance_cross_entropy(logits, target, mask)
        + torsion_angle_loss(pred, target, mask)
        + inter_residue_geometry_loss(pred, mask)
    )
    loss.backward()

    c1 = RNA_ATOM_NAMES.index("C1'")
    assert torch.isfinite(loss)
    assert pred.grad is not None
    assert pred.grad[:, :, c1].abs().sum() > 0
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_single_c1_labels_train_contact_head_without_residue_frames():
    target = torch.tensor([[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [15.0, 0.0, 0.0]]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    contact = torch.zeros(1, 3, 3, 1, requires_grad=True)

    loss = pair_orientation_cross_entropy({"contact": contact}, target, mask)
    loss.backward()

    assert loss.item() > 0
    assert contact.grad is not None
    assert contact.grad.abs().sum().item() > 0


def test_small_model_can_overfit_one_structure_without_coordinate_collapse():
    config = RhoFoldConfig(
        d_model=16,
        pair_dim=8,
        msa_dim=8,
        nhead=4,
        pair_heads=2,
        num_e2e_layers=1,
        num_structure_layers=1,
        dim_feedforward=32,
        recycle_iters=1,
        random_recycle_training=False,
        dropout=0.0,
    )
    input_ids = torch.tensor([[1, 2, 3, 4, 1, 2]])
    torch.manual_seed(31)
    teacher = RhoFoldModel(config).eval()
    with torch.no_grad():
        teacher.structure_module.torsion_head.bias.add_(
            torch.linspace(-0.35, 0.35, 14)
        )
        teacher.structure_module.sugar_pucker_head.bias.add_(
            torch.tensor([0.2, -0.1])
        )
    with torch.inference_mode():
        target = teacher(input_ids)
    torch.manual_seed(32)
    model = RhoFoldModel(config).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    mask = torch.ones(target.shape[:-1], dtype=torch.bool)
    losses = []
    for _ in range(21):
        pred = model(input_ids)
        loss = (
            kabsch_aligned_coordinate_loss(pred, target, mask)
            + 0.2 * masked_pairwise_distance_mse(pred, target, mask)
        )
        losses.append(float(loss.detach()))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    assert losses[-1] < 0.4 * losses[0]
    c1 = RNA_ATOM_NAMES.index("C1'")
    adjacent = torch.linalg.norm(pred[0, 1:, c1] - pred[0, :-1, c1], dim=-1)
    assert adjacent.mean() > 2.0
