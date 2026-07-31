from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from rna_scaffold_3d.geometry import build_residue_frames, kabsch_align
from rna_scaffold_3d.rna_atoms import (
    RNA_ATOM_TO_INDEX,
    RNA_BASE_BOND_LENGTHS,
    RNA_COMMON_BOND_LENGTHS,
    chemical_atom_mask,
    chemical_bond_adjacency,
)


# ——— van der Waals radii for RNA heavy atoms (Å) ———

_VDW_RADII = {
    "P": 1.80,
    "O": 1.52,
    "N": 1.55,
    "C": 1.70,
}

# classify each of the 27 RNA atoms by element
_ATOM_ELEMENTS = [
    "P",   # P
    "O",   # OP1
    "O",   # OP2
    "O",   # O5'
    "C",   # C5'
    "C",   # C4'
    "O",   # O4'
    "C",   # C3'
    "O",   # O3'
    "C",   # C2'
    "O",   # O2'
    "C",   # C1'
    "N",   # N1
    "C",   # C2
    "O",   # O2
    "N",   # N2
    "N",   # N3
    "C",   # C4
    "N",   # N4
    "C",   # C5
    "C",   # C6
    "O",   # O4
    "N",   # N9
    "C",   # C8
    "N",   # N7
    "N",   # N6
    "O",   # O6
]

_ATOM_VDW = torch.tensor([_VDW_RADII[element] for element in _ATOM_ELEMENTS], dtype=torch.float32)


# ——— core losses ———


def _masked_mean_per_example(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Average valid elements within each example, then average examples."""
    expanded_mask = mask.bool()
    while expanded_mask.ndim < values.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    expanded_mask = expanded_mask.expand_as(values)
    flat_values = values.reshape(values.size(0), -1)
    flat_mask = expanded_mask.reshape(values.size(0), -1)
    counts = flat_mask.sum(dim=-1)
    valid_examples = counts.gt(0)
    if not valid_examples.any():
        return values.sum() * 0.0
    per_example = (
        (flat_values * flat_mask.to(values.dtype)).sum(dim=-1)
        / counts.clamp(min=1).to(values.dtype)
    )
    return per_example[valid_examples].mean()


def _three_point_frames(
    coords: torch.Tensor,
    mask: torch.Tensor,
    origin_index: int,
    x_index: int,
    plane_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    origin = coords[..., origin_index, :]
    x_vector = coords[..., x_index, :] - origin
    plane_vector = coords[..., plane_index, :] - origin
    x_norm = torch.linalg.norm(x_vector, dim=-1)
    x_axis = x_vector / x_norm.clamp_min(1e-8).unsqueeze(-1)
    z_vector = torch.linalg.cross(x_axis, plane_vector, dim=-1)
    z_norm = torch.linalg.norm(z_vector, dim=-1)
    z_axis = z_vector / z_norm.clamp_min(1e-8).unsqueeze(-1)
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    frame = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    valid = (
        mask[..., origin_index]
        & mask[..., x_index]
        & mask[..., plane_index]
        & x_norm.gt(1e-6)
        & z_norm.gt(1e-6)
    )
    return frame, valid


def base_relative_orientation_targets(
    coords: torch.Tensor,
    coord_mask: torch.Tensor,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build global-rigid-invariant sugar-to-base rotations."""
    if coords.ndim != 4 or coord_mask.shape != coords.shape[:-1]:
        raise ValueError("Expected coords [B,L,A,3] and matching coord_mask.")
    if input_ids.shape != coords.shape[:2]:
        raise ValueError("input_ids must match coords batch and length.")
    sugar_spec = tuple(
        RNA_ATOM_TO_INDEX[name] for name in ("C1'", "C2'", "O4'")
    )
    sugar_frame, sugar_valid = _three_point_frames(
        coords, coord_mask, *sugar_spec
    )
    relative = coords.new_zeros((*coords.shape[:2], 3, 3))
    valid = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id, names in {
        1: ("N9", "C4", "C8"),
        2: ("N1", "C2", "C6"),
        3: ("N1", "C2", "C6"),
        4: ("N9", "C4", "C8"),
    }.items():
        base_spec = tuple(RNA_ATOM_TO_INDEX[name] for name in names)
        base_frame, base_valid = _three_point_frames(
            coords, coord_mask, *base_spec
        )
        residue_valid = input_ids.eq(token_id) & sugar_valid & base_valid
        residue_relative = sugar_frame.transpose(-2, -1) @ base_frame
        relative = torch.where(
            residue_valid[..., None, None], residue_relative, relative
        )
        valid |= residue_valid
    return relative, valid


def base_orientation_coordinate_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    coord_mask: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """SO(3) cosine loss for final sugar-relative base orientation."""
    predicted_rotations, predicted_valid = base_relative_orientation_targets(
        pred, chemical_atom_mask(input_ids), input_ids
    )
    target_rotations, target_valid = base_relative_orientation_targets(
        target, coord_mask.bool(), input_ids
    )
    valid = predicted_valid & target_valid
    difference = (
        predicted_rotations.transpose(-2, -1) @ target_rotations
    )
    cosine = (
        difference.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0
    ) / 2.0
    return _masked_mean_per_example(
        1.0 - cosine.clamp(-1.0, 1.0), valid
    )


def masked_coordinate_mse(pred: torch.Tensor, target: torch.Tensor, coord_mask: torch.Tensor) -> torch.Tensor:
    if pred.ndim == 4 and target.ndim == 3:
        pred = pred[..., RNA_ATOM_TO_INDEX["C1'"], :]
    return _masked_mean_per_example((pred - target).pow(2), coord_mask)


def masked_coordinate_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    coord_mask: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    if pred.ndim == 4 and target.ndim == 3:
        pred = pred[..., RNA_ATOM_TO_INDEX["C1'"], :]
    element_loss = F.smooth_l1_loss(
        pred, target, beta=beta, reduction="none"
    )
    return _masked_mean_per_example(element_loss, coord_mask)


def kabsch_aligned_coordinate_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    coord_mask: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Coordinate loss after optimal rigid alignment of valid atoms."""
    if pred.ndim == 4 and target.ndim == 3:
        pred = pred[..., RNA_ATOM_TO_INDEX["C1'"], :]
    if pred.ndim != target.ndim:
        raise ValueError("pred and target ranks are incompatible for Kabsch alignment.")
    flat_pred = pred.reshape(pred.size(0), -1, 3)
    flat_target = target.reshape(target.size(0), -1, 3)
    flat_mask = coord_mask.reshape(coord_mask.size(0), -1)
    aligned = kabsch_align(flat_pred, flat_target, flat_mask)
    if not flat_mask.any():
        return pred.sum() * 0.0
    absolute_error = (aligned - flat_target).abs()
    per_coordinate = (
        torch.where(
            absolute_error < beta,
            0.5 * absolute_error.square() / max(beta, 1e-12),
            absolute_error - 0.5 * beta,
        )
        if beta > 0
        else absolute_error
    ).sum(dim=-1)
    counts = flat_mask.sum(dim=-1)
    valid_examples = counts.gt(0)
    per_example = (
        (per_coordinate * flat_mask).sum(dim=-1)
        / (counts.clamp(min=1) * 3).to(per_coordinate.dtype)
    )
    return per_example[valid_examples].mean()


def masked_pairwise_distance_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    coord_mask: torch.Tensor,
    distance_scale: float = 10.0,
    clamp_distance: float = 30.0,
) -> torch.Tensor:
    """Robust, normalized C1' distance-map loss (legacy function name retained)."""
    losses: list[torch.Tensor] = []
    if pred.ndim == 4:
        representative = RNA_ATOM_TO_INDEX["C1'"] if pred.size(-2) > RNA_ATOM_TO_INDEX["C1'"] else 0
        pred = pred[..., representative, :]
        if target.ndim == 4:
            target = target[..., representative, :]
            coord_mask = coord_mask[..., representative]
    for pred_item, target_item, mask_item in zip(pred, target, coord_mask):
        if int(mask_item.sum().item()) < 2:
            continue
        pred_valid = pred_item[mask_item]
        target_valid = target_item[mask_item]
        pred_dist = torch.cdist(pred_valid, pred_valid)
        target_dist = torch.cdist(target_valid, target_valid)
        difference = (pred_dist - target_dist).clamp(
            min=-clamp_distance,
            max=clamp_distance,
        ) / distance_scale
        losses.append(F.smooth_l1_loss(difference, torch.zeros_like(difference), beta=0.1))
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def local_distance_difference_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    coord_mask: torch.Tensor,
    cutoff: float = 15.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Smooth lDDT-aligned loss over native-local C1' residue pairs."""
    if pred.ndim == 4:
        c1 = RNA_ATOM_TO_INDEX["C1'"] if pred.size(-2) > RNA_ATOM_TO_INDEX["C1'"] else 0
        pred = pred[..., c1, :]
        if target.ndim == 4:
            target = target[..., c1, :]
            coord_mask = coord_mask[..., c1]
    losses: list[torch.Tensor] = []
    for predicted, observed, valid in zip(pred, target, coord_mask):
        if int(valid.sum().item()) < 2:
            continue
        predicted = predicted[valid].float()
        observed = observed[valid].float()
        pred_distance = torch.cdist(predicted, predicted)
        target_distance = torch.cdist(observed, observed)
        identity = torch.eye(
            predicted.size(0), dtype=torch.bool, device=pred.device
        )
        pair_mask = ~identity & target_distance.lt(cutoff)
        if not pair_mask.any():
            continue
        losses.append(
            F.smooth_l1_loss(
                pred_distance[pair_mask],
                target_distance[pair_mask],
                beta=beta,
            )
        )
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean().to(pred.dtype)


def soft_lddt_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    coord_mask: torch.Tensor,
    cutoff: float = 15.0,
    temperature: float = 0.25,
) -> torch.Tensor:
    """Differentiable C1'-lDDT objective with per-example normalization."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if pred.ndim == 4:
        c1 = (
            RNA_ATOM_TO_INDEX["C1'"]
            if pred.size(-2) > RNA_ATOM_TO_INDEX["C1'"]
            else 0
        )
        pred = pred[..., c1, :]
        if target.ndim == 4:
            target = target[..., c1, :]
            coord_mask = coord_mask[..., c1]
    thresholds = pred.new_tensor((0.5, 1.0, 2.0, 4.0))
    ideal_score = torch.sigmoid(thresholds / temperature).mean()
    losses: list[torch.Tensor] = []
    for predicted, observed, valid in zip(pred, target, coord_mask):
        if int(valid.sum().item()) < 2:
            continue
        predicted = predicted[valid].float()
        observed = observed[valid].float()
        predicted_distance = torch.cdist(predicted, predicted)
        observed_distance = torch.cdist(observed, observed)
        identity = torch.eye(
            predicted.size(0), dtype=torch.bool, device=pred.device
        )
        neighborhood = ~identity & observed_distance.lt(cutoff)
        if not neighborhood.any():
            continue
        error = (predicted_distance - observed_distance).abs()
        score = torch.sigmoid(
            (
                thresholds.float().view(-1, 1, 1)
                - error.unsqueeze(0)
            )
            / temperature
        ).mean(dim=0)
        normalized_score = score[neighborhood].mean() / ideal_score.float()
        losses.append(1.0 - normalized_score.clamp(max=1.0))
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean().to(pred.dtype)


def pair_distance_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    coord_mask: torch.Tensor,
    bin_edges: torch.Tensor | None = None,
    atom_index: int = RNA_ATOM_TO_INDEX["C1'"],
) -> torch.Tensor:
    if bin_edges is None:
        bin_edges = torch.linspace(2.0, 40.0, logits.size(-1) - 1, device=logits.device)
    else:
        bin_edges = bin_edges.to(logits.device)
    if target.ndim == 4:
        atom_index = atom_index if target.size(-2) > atom_index else 0
        target_points = target[..., atom_index, :]
        point_mask = coord_mask[..., atom_index]
    else:
        target_points = target
        point_mask = coord_mask
    losses: list[torch.Tensor] = []
    for item_logits, points, mask in zip(logits, target_points, point_mask):
        if int(mask.sum().item()) < 2:
            continue
        pair_mask = mask.unsqueeze(0) & mask.unsqueeze(1)
        distances = torch.cdist(points, points)
        bins = torch.bucketize(distances, bin_edges)
        valid_logits = item_logits[pair_mask]
        valid_bins = bins[pair_mask]
        losses.append(F.cross_entropy(valid_logits, valid_bins))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def pair_orientation_cross_entropy(
    logits: dict[str, torch.Tensor],
    target: torch.Tensor,
    coord_mask: torch.Tensor,
    distance_cutoff: float = 30.0,
    input_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervise directed omega/theta/phi channels and a symmetric contact head."""
    losses: list[torch.Tensor] = []
    diagonal = torch.eye(target.size(1), dtype=torch.bool, device=target.device).unsqueeze(0)
    if target.ndim == 4 and coord_mask.ndim == 3:
        rotations, origins, frame_mask = build_residue_frames(
            target, coord_mask, input_ids
        )
        relative = origins.unsqueeze(1) - origins.unsqueeze(2)
        distance = torch.linalg.norm(relative, dim=-1)
        pair_mask = (
            frame_mask.unsqueeze(1)
            & frame_mask.unsqueeze(2)
            & ~diagonal
            & distance.lt(distance_cutoff)
        )
        if pair_mask.any():
            # R_i^T (x_j - x_i), with local basis vectors stored as columns.
            local_direction = torch.einsum("blji,blmj->blmi", rotations, relative)
            theta = torch.atan2(local_direction[..., 1], local_direction[..., 0])
            phi = torch.acos(
                (local_direction[..., 2] / distance.clamp(min=1e-6)).clamp(-1.0, 1.0)
            )
            e1 = rotations[..., :, 0]
            local_e1_j = torch.einsum("blji,bmj->blmi", rotations, e1)
            omega = torch.atan2(local_e1_j[..., 1], local_e1_j[..., 0])
            for name, angle, period in (
                ("omega", omega, 2 * torch.pi),
                ("theta", theta, 2 * torch.pi),
                ("phi", phi, torch.pi),
            ):
                if name not in logits:
                    continue
                channels = logits[name].size(-1)
                offset = torch.pi if period == 2 * torch.pi else 0.0
                normalized = torch.remainder(angle + offset, period)
                bins = torch.floor(normalized / period * channels).long().clamp(max=channels - 1)
                element_loss = F.cross_entropy(
                    logits[name].movedim(-1, 1),
                    bins,
                    reduction="none",
                )
                losses.append(
                    _masked_mean_per_example(element_loss, pair_mask)
                )
    if "contact" in logits:
        if target.ndim == 4 and coord_mask.ndim == 3:
            c1 = RNA_ATOM_TO_INDEX["C1'"]
            contact_points = target[..., c1, :]
            contact_point_mask = coord_mask[..., c1]
        elif target.ndim == 3 and coord_mask.ndim == 2:
            contact_points = target
            contact_point_mask = coord_mask
        else:
            contact_points = None
            contact_point_mask = None
        if contact_points is not None:
            c1_distance = torch.cdist(contact_points.float(), contact_points.float())
            contact_mask = (
                contact_point_mask.unsqueeze(1)
                & contact_point_mask.unsqueeze(2)
                & ~diagonal
            )
        else:
            contact_mask = torch.zeros_like(diagonal)
            c1_distance = None
        if contact_mask.any():
            contact_target = c1_distance.lt(8.0).to(logits["contact"].dtype)
            contact_logits = logits["contact"].squeeze(-1)
            element_loss = F.binary_cross_entropy_with_logits(
                contact_logits, contact_target, reduction="none"
            )
            losses.append(
                _masked_mean_per_example(element_loss, contact_mask)
            )
    return torch.stack(losses).mean() if losses else next(iter(logits.values())).sum() * 0.0


_BOND_SPECS = (
    ("P", "O5'", 1.60),
    ("O5'", "C5'", 1.40),
    ("C5'", "C4'", 1.50),
    ("C4'", "C3'", 1.50),
    ("C3'", "O3'", 1.40),
)

_ANGLE_SPECS = (
    ("P", "O5'", "C5'", 120.0),
    ("O5'", "C5'", "C4'", 110.0),
    ("C5'", "C4'", "C3'", 116.0),
    ("C4'", "C3'", "O3'", 113.0),
)

_GLYCOSIDIC_ANGLE_SPECS = {
    "A": (("O4'", "C1'", "N9", 108.7), ("C2'", "C1'", "N9", 112.6)),
    "U": (("O4'", "C1'", "N1", 108.7), ("C2'", "C1'", "N1", 112.6)),
    "C": (("O4'", "C1'", "N1", 108.7), ("C2'", "C1'", "N1", 112.6)),
    "G": (("O4'", "C1'", "N9", 108.7), ("C2'", "C1'", "N9", 112.6)),
}

_TORSION_SPECS = (
    ("alpha", ("O3'", -1), ("P", 0), ("O5'", 0), ("C5'", 0)),
    ("beta", ("P", 0), ("O5'", 0), ("C5'", 0), ("C4'", 0)),
    ("gamma", ("O5'", 0), ("C5'", 0), ("C4'", 0), ("C3'", 0)),
    ("delta", ("C5'", 0), ("C4'", 0), ("C3'", 0), ("O3'", 0)),
    ("epsilon", ("C4'", 0), ("C3'", 0), ("O3'", 0), ("P", 1)),
    ("zeta", ("C3'", 0), ("O3'", 0), ("P", 1), ("O5'", 1)),
    ("chi_pyrimidine", ("O4'", 0), ("C1'", 0), ("N1", 0), ("C2", 0)),
    ("chi_purine", ("O4'", 0), ("C1'", 0), ("N9", 0), ("C4", 0)),
)


def bond_length_loss(
    pred: torch.Tensor,
    coord_mask: torch.Tensor,
    input_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if pred.ndim != 4 or coord_mask.ndim != 3:
        return pred.sum() * 0.0
    values: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []

    def add_specs(
        specs: tuple[tuple[str, str, float], ...],
        residue_mask: torch.Tensor | None = None,
    ) -> None:
        for atom_a, atom_b, target_length in specs:
            idx_a = RNA_ATOM_TO_INDEX[atom_a]
            idx_b = RNA_ATOM_TO_INDEX[atom_b]
            valid = coord_mask[..., idx_a] & coord_mask[..., idx_b]
            if residue_mask is not None:
                valid &= residue_mask
            distances = torch.linalg.norm(
                pred[..., idx_a, :] - pred[..., idx_b, :], dim=-1
            )
            values.append((distances - target_length).pow(2))
            masks.append(valid)

    if input_ids is None:
        add_specs(_BOND_SPECS)
    else:
        known_residue = input_ids.ge(1) & input_ids.le(4)
        add_specs(RNA_COMMON_BOND_LENGTHS, known_residue)
        for token_id, base in enumerate(("A", "U", "C", "G"), start=1):
            add_specs(RNA_BASE_BOND_LENGTHS[base], input_ids.eq(token_id))
    if not values:
        return pred.sum() * 0.0
    return _masked_mean_per_example(
        torch.stack(values, dim=1), torch.stack(masks, dim=1)
    )


def bond_angle_loss(
    pred: torch.Tensor,
    coord_mask: torch.Tensor,
    input_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if pred.ndim != 4 or coord_mask.ndim != 3:
        return pred.sum() * 0.0
    values: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for atom_a, atom_b, atom_c, target_degrees in _ANGLE_SPECS:
        idx_a = RNA_ATOM_TO_INDEX[atom_a]
        idx_b = RNA_ATOM_TO_INDEX[atom_b]
        idx_c = RNA_ATOM_TO_INDEX[atom_c]
        valid = coord_mask[..., idx_a] & coord_mask[..., idx_b] & coord_mask[..., idx_c]
        ba = pred[..., idx_a, :] - pred[..., idx_b, :]
        bc = pred[..., idx_c, :] - pred[..., idx_b, :]
        cos_angle = F.cosine_similarity(ba, bc, dim=-1).clamp(
            -1.0 + 1e-6, 1.0 - 1e-6
        )
        angle = torch.rad2deg(torch.acos(cos_angle))
        values.append((angle - target_degrees).pow(2) / 100.0)
        masks.append(valid)
    if input_ids is not None:
        for token_id, base in enumerate(("A", "U", "C", "G"), start=1):
            for atom_a, atom_b, atom_c, target_degrees in (
                _GLYCOSIDIC_ANGLE_SPECS[base]
            ):
                idx_a = RNA_ATOM_TO_INDEX[atom_a]
                idx_b = RNA_ATOM_TO_INDEX[atom_b]
                idx_c = RNA_ATOM_TO_INDEX[atom_c]
                valid = (
                    coord_mask[..., idx_a]
                    & coord_mask[..., idx_b]
                    & coord_mask[..., idx_c]
                    & input_ids.eq(token_id)
                )
                ba = pred[..., idx_a, :] - pred[..., idx_b, :]
                bc = pred[..., idx_c, :] - pred[..., idx_b, :]
                cos_angle = F.cosine_similarity(
                    ba, bc, dim=-1
                ).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
                angle = torch.rad2deg(torch.acos(cos_angle))
                values.append(
                    (angle - target_degrees).pow(2) / 100.0
                )
                masks.append(valid)
    return _masked_mean_per_example(
        torch.stack(values, dim=1), torch.stack(masks, dim=1)
    )


def torsion_angle_loss(
    pred: torch.Tensor,
    target: torch.Tensor | None,
    coord_mask: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Periodic RNA alpha..zeta/chi loss against observed target torsions."""
    if target is None or coord_mask is None:
        raise ValueError(
            "torsion_angle_loss requires predicted coordinates, target "
            "coordinates, and an observation mask."
        )
    if pred.ndim == 4 and target.ndim == 3 and coord_mask.ndim == 2:
        # A single representative atom per residue cannot define RNA
        # torsions. This is a valid sparse-label dataset, not malformed input.
        return pred.sum() * 0.0
    if pred.ndim != 4 or target.ndim != 4 or coord_mask.ndim != 3:
        raise ValueError(
            "torsion_angle_loss expects [B,L,A,3] coordinates and "
            "[B,L,A] coord_mask."
        )
    values: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    length = pred.size(1)
    residue_positions = torch.arange(length, device=pred.device)
    for torsion_name, *atoms in _TORSION_SPECS:
        atom_indices = [RNA_ATOM_TO_INDEX[atom] for atom, _ in atoms]
        offsets = [offset for _, offset in atoms]
        shifted = [residue_positions + offset for offset in offsets]
        valid_position = torch.ones(length, dtype=torch.bool, device=pred.device)
        for position in shifted:
            valid_position &= (position >= 0) & (position < length)
        safe_positions = [position.clamp(0, length - 1) for position in shifted]
        points_pred = [
            pred[:, position, atom_index, :]
            for position, atom_index in zip(safe_positions, atom_indices)
        ]
        points_target = [
            target[:, position, atom_index, :]
            for position, atom_index in zip(safe_positions, atom_indices)
        ]
        valid = valid_position.unsqueeze(0).expand(pred.size(0), -1).clone()
        for position, atom_index in zip(safe_positions, atom_indices):
            valid &= coord_mask[:, position, atom_index]
        if torsion_name == "chi_purine":
            selector = (
                input_ids.eq(1) | input_ids.eq(4)
                if input_ids is not None
                else coord_mask[..., RNA_ATOM_TO_INDEX["N9"]]
            )
            valid &= selector
        elif torsion_name == "chi_pyrimidine":
            selector = (
                input_ids.eq(2) | input_ids.eq(3)
                if input_ids is not None
                else ~coord_mask[..., RNA_ATOM_TO_INDEX["N9"]]
            )
            valid &= selector
        pred_angle = _dihedral(*points_pred)
        target_angle = _dihedral(*points_target)
        values.append(1.0 - torch.cos(pred_angle - target_angle))
        masks.append(valid)
    return _masked_mean_per_example(
        torch.stack(values, dim=1), torch.stack(masks, dim=1)
    )


def rna_torsion_targets(
    coords: torch.Tensor,
    coord_mask: torch.Tensor,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract alpha..zeta/chi targets as normalized (sin, cos) pairs."""
    if coords.ndim != 4 or coord_mask.ndim != 3:
        raise ValueError(
            "coords and coord_mask must have shapes [B, L, A, 3] and [B, L, A]."
        )
    if input_ids is not None and input_ids.shape != coords.shape[:2]:
        raise ValueError("input_ids must match the batch/residue dimensions.")
    batch, length = coords.shape[:2]
    residue_positions = torch.arange(length, device=coords.device)
    target_angles = coords.new_zeros(batch, length, 7)
    target_mask = torch.zeros(
        batch, length, 7, dtype=torch.bool, device=coords.device
    )
    for spec_index, (torsion_name, *atoms) in enumerate(_TORSION_SPECS):
        channel = min(spec_index, 6)
        atom_indices = [RNA_ATOM_TO_INDEX[atom] for atom, _ in atoms]
        shifted = [
            residue_positions + offset for _, offset in atoms
        ]
        valid_position = torch.ones(
            length, dtype=torch.bool, device=coords.device
        )
        for position in shifted:
            valid_position &= (position >= 0) & (position < length)
        safe_positions = [
            position.clamp(0, length - 1) for position in shifted
        ]
        points = [
            coords[:, position, atom_index, :]
            for position, atom_index in zip(safe_positions, atom_indices)
        ]
        valid = valid_position.unsqueeze(0).expand(batch, -1).clone()
        for position, atom_index in zip(safe_positions, atom_indices):
            valid &= coord_mask[:, position, atom_index]
        if torsion_name == "chi_purine":
            selector = (
                input_ids.eq(1) | input_ids.eq(4)
                if input_ids is not None
                else coord_mask[..., RNA_ATOM_TO_INDEX["N9"]]
            )
            valid &= selector
        elif torsion_name == "chi_pyrimidine":
            selector = (
                input_ids.eq(2) | input_ids.eq(3)
                if input_ids is not None
                else ~coord_mask[..., RNA_ATOM_TO_INDEX["N9"]]
            )
            valid &= selector
        angle = _dihedral(*points)
        target_angles[..., channel] = torch.where(
            valid, angle, target_angles[..., channel]
        )
        target_mask[..., channel] |= valid
    targets = torch.stack(
        (torch.sin(target_angles), torch.cos(target_angles)), dim=-1
    )
    return targets, target_mask


def torsion_parameter_loss(
    predicted_torsions: torch.Tensor,
    target_coords: torch.Tensor,
    coord_mask: torch.Tensor,
    input_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Direct periodic loss on the final refined alpha..zeta/chi parameters."""
    if predicted_torsions.ndim != 4 or predicted_torsions.shape[-2:] != (7, 2):
        return predicted_torsions.sum() * 0.0
    targets, valid = rna_torsion_targets(
        target_coords, coord_mask, input_ids
    )
    predicted = F.normalize(predicted_torsions, dim=-1, eps=1e-6)
    periodic_error = 1.0 - (predicted * targets).sum(dim=-1)
    return _masked_mean_per_example(periodic_error, valid)


def sugar_pucker_phase(
    coords: torch.Tensor,
    coord_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target ribose pseudorotation phase as normalized (sin P, cos P)."""
    ring = ("O4'", "C1'", "C2'", "C3'", "C4'")
    indices = [RNA_ATOM_TO_INDEX[name] for name in ring]
    valid = torch.ones(coords.shape[:2], dtype=torch.bool, device=coords.device)
    for atom_index in indices:
        valid &= coord_mask[..., atom_index]
    points = [coords[..., atom_index, :] for atom_index in indices]
    # ν0..ν4 follow consecutive endocyclic torsions around the furanose ring.
    nu0 = _dihedral(points[0], points[1], points[2], points[3])
    nu1 = _dihedral(points[1], points[2], points[3], points[4])
    nu2 = _dihedral(points[2], points[3], points[4], points[0])
    nu3 = _dihedral(points[3], points[4], points[0], points[1])
    nu4 = _dihedral(points[4], points[0], points[1], points[2])
    numerator = nu4 + nu1 - nu3 - nu0
    denominator = 2.0 * nu2 * (
        math.sin(math.radians(36.0)) + math.sin(math.radians(72.0))
    )
    # Our dihedral ordering starts the five-membered ring one position later
    # than the conventional Altona-Sundaralingam phase definition. Correct
    # that cyclic index shift (4*pi/5 = 144 degrees) so C3'-endo is near 18
    # degrees and C2'-endo is near 162 degrees.
    phase = torch.atan2(numerator, denominator) - 4.0 * math.pi / 5.0
    phase_sin_cos = torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)
    finite = torch.isfinite(phase_sin_cos).all(dim=-1)
    return phase_sin_cos, valid & finite


def sugar_pucker_phase_loss(
    predicted_phase: torch.Tensor,
    target: torch.Tensor,
    coord_mask: torch.Tensor,
) -> torch.Tensor:
    """Periodic 1-cos loss for the explicitly predicted sugar pucker phase."""
    if predicted_phase.ndim != 3 or target.ndim != 4 or coord_mask.ndim != 3:
        return predicted_phase.sum() * 0.0
    target_phase, valid = sugar_pucker_phase(target, coord_mask)
    predicted_phase = F.normalize(predicted_phase, dim=-1, eps=1e-6)
    cosine_delta = (predicted_phase * target_phase).sum(dim=-1)
    return _masked_mean_per_example(1.0 - cosine_delta, valid)


def sugar_pucker_coordinate_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    coord_mask: torch.Tensor,
) -> torch.Tensor:
    """Periodic pseudorotation loss measured on final generated coordinates."""
    if pred.ndim != 4 or target.ndim != 4 or coord_mask.ndim != 3:
        return pred.sum() * 0.0
    predicted_phase, predicted_valid = sugar_pucker_phase(pred, coord_mask)
    target_phase, target_valid = sugar_pucker_phase(target, coord_mask)
    valid = predicted_valid & target_valid
    cosine_delta = (predicted_phase * target_phase).sum(dim=-1)
    return _masked_mean_per_example(1.0 - cosine_delta, valid)


def inter_residue_geometry_loss(
    pred: torch.Tensor,
    coord_mask: torch.Tensor,
) -> torch.Tensor:
    """Phosphodiester O3'(i)-P(i+1) bond and adjacent bond-angle restraints."""
    if pred.ndim != 4 or coord_mask.ndim != 3 or pred.size(1) < 2:
        return pred.sum() * 0.0
    o3 = RNA_ATOM_TO_INDEX["O3'"]
    p = RNA_ATOM_TO_INDEX["P"]
    o5 = RNA_ATOM_TO_INDEX["O5'"]
    c3 = RNA_ATOM_TO_INDEX["C3'"]
    bond_valid = coord_mask[:, :-1, o3] & coord_mask[:, 1:, p]
    values: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    distance = torch.linalg.norm(pred[:, :-1, o3] - pred[:, 1:, p], dim=-1)
    values.append((distance - 1.60).pow(2))
    masks.append(bond_valid)
    angle_specs = (
        (c3, 0, o3, 0, p, 1, 120.0),
        (o3, 0, p, 1, o5, 1, 104.0),
    )
    for a, oa, b, ob, c, oc, expected in angle_specs:
        left = slice(0, -1) if oa == 0 else slice(1, None)
        middle = slice(0, -1) if ob == 0 else slice(1, None)
        right = slice(0, -1) if oc == 0 else slice(1, None)
        valid = coord_mask[:, left, a] & coord_mask[:, middle, b] & coord_mask[:, right, c]
        ba = pred[:, left, a] - pred[:, middle, b]
        bc = pred[:, right, c] - pred[:, middle, b]
        angle = torch.rad2deg(torch.acos(F.cosine_similarity(ba, bc, dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)))
        values.append((angle - expected).pow(2) / 100.0)
        masks.append(valid)
    return _masked_mean_per_example(
        torch.stack(values, dim=1), torch.stack(masks, dim=1)
    )


def base_planarity_loss(pred: torch.Tensor, coord_mask: torch.Tensor) -> torch.Tensor:
    """Keep observed nucleobase atoms close to the plane defined by N/C atoms."""
    if pred.ndim != 4 or coord_mask.ndim != 3:
        return pred.sum() * 0.0
    base_indices = [RNA_ATOM_TO_INDEX[name] for name in ("N1", "C2", "N3", "C4", "C5", "C6", "N9", "C8", "N7")]
    values: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for anchor_names in (("N1", "C2", "C4"), ("N9", "C8", "C4")):
        anchors = [RNA_ATOM_TO_INDEX[name] for name in anchor_names]
        valid_anchor = coord_mask[..., anchors].all(dim=-1)
        a, b, c = (pred[..., index, :] for index in anchors)
        normal = F.normalize(torch.cross(b - a, c - a, dim=-1), dim=-1, eps=1e-6)
        for index in base_indices:
            valid = valid_anchor & coord_mask[..., index]
            distance = ((pred[..., index, :] - a) * normal).sum(dim=-1).abs()
            values.append(distance.pow(2))
            masks.append(valid)
    return _masked_mean_per_example(
        torch.stack(values, dim=1), torch.stack(masks, dim=1)
    )


def sugar_ring_closure_loss(pred: torch.Tensor, coord_mask: torch.Tensor) -> torch.Tensor:
    """Constrain the two closing bonds of the ribose ring."""
    specs = (("O4'", "C1'", 1.43), ("O4'", "C4'", 1.45), ("C1'", "C2'", 1.53), ("C2'", "C3'", 1.53))
    if pred.ndim != 4 or coord_mask.ndim != 3:
        return pred.sum() * 0.0
    values: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for name_a, name_b, expected in specs:
        a, b = RNA_ATOM_TO_INDEX[name_a], RNA_ATOM_TO_INDEX[name_b]
        valid = coord_mask[..., a] & coord_mask[..., b]
        distance = torch.linalg.norm(pred[..., a, :] - pred[..., b, :], dim=-1)
        values.append((distance - expected).pow(2))
        masks.append(valid)
    return _masked_mean_per_example(
        torch.stack(values, dim=1), torch.stack(masks, dim=1)
    )


def plddt_confidence_loss(
    predicted_plddt: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    coord_mask: torch.Tensor,
    cutoff: float = 15.0,
) -> torch.Tensor:
    """Train confidence against a rigid-transform-invariant C1'-lDDT target."""
    if pred.ndim != 4 or target.ndim != 4:
        return predicted_plddt.sum() * 0.0
    representative = RNA_ATOM_TO_INDEX["C1'"]
    residue_mask = coord_mask[..., representative]
    if not residue_mask.any():
        return predicted_plddt.sum() * 0.0
    pred_distance = torch.cdist(pred[..., representative, :].float(), pred[..., representative, :].float())
    target_distance = torch.cdist(target[..., representative, :].float(), target[..., representative, :].float())
    pair_mask = residue_mask.unsqueeze(1) & residue_mask.unsqueeze(2)
    identity = torch.eye(pred.size(1), dtype=torch.bool, device=pred.device).unsqueeze(0)
    neighborhood = pair_mask & ~identity & (target_distance < cutoff)
    distance_error = (pred_distance - target_distance).abs()
    scores = torch.stack([(distance_error < threshold).float() for threshold in (0.5, 1.0, 2.0, 4.0)])
    scores = scores.mean(dim=0)
    counts = neighborhood.sum(dim=-1)
    target_confidence = (
        (scores * neighborhood).sum(dim=-1) / counts.clamp(min=1)
    ).detach() * 100.0
    residue_mask = residue_mask & counts.gt(0)
    if not residue_mask.any():
        return predicted_plddt.sum() * 0.0
    confidence_error = F.smooth_l1_loss(
        predicted_plddt, target_confidence, beta=5.0, reduction="none"
    )
    return _masked_mean_per_example(confidence_error, residue_mask)


def _dihedral(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = F.normalize(b1, dim=-1)
    v = b0 - (b0 * b1).sum(dim=-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(dim=-1, keepdim=True) * b1
    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1, v, dim=-1) * w).sum(dim=-1)
    degenerate = (x.square() + y.square()).detach().lt(1e-12)
    safe_x = torch.where(degenerate, torch.ones_like(x), x)
    safe_y = torch.where(degenerate, torch.zeros_like(y), y)
    return torch.atan2(safe_y, safe_x)


# ——— steric clash penalty ———


def steric_clash_loss(
    pred: torch.Tensor,
    coord_mask: torch.Tensor,
    input_ids: torch.Tensor | None = None,
    clash_tolerance: float = 0.6,
    residue_chunk_size: int = 4,
) -> torch.Tensor:
    """Chunked all-atom clash loss with RNA covalent/1–3 exclusions.

    Supplying target ``input_ids`` enables chemically aware same-residue
    clashes. The legacy two-argument form retains inter-residue behavior for
    callers that do not know residue identity.
    """
    if pred.ndim != 4 or coord_mask.ndim != 3:
        return pred.sum() * 0.0
    vdw = _ATOM_VDW.to(device=pred.device, dtype=pred.dtype)
    minimum_distance = (
        vdw.view(1, 1, -1, 1)
        + vdw.view(1, 1, 1, -1)
        - clash_tolerance
    )
    example_losses: list[torch.Tensor] = []
    o3 = RNA_ATOM_TO_INDEX["O3'"]
    p = RNA_ATOM_TO_INDEX["P"]
    for item_index, (item_coords, item_mask) in enumerate(zip(pred, coord_mask)):
        length = item_coords.size(0)
        item_penalty = item_coords.sum() * 0.0
        item_pair_count = torch.zeros((), device=pred.device, dtype=torch.long)
        residue_indices = torch.arange(length, device=pred.device)
        atom_indices = torch.arange(item_coords.size(1), device=pred.device)
        intra_excluded = None
        if input_ids is not None:
            adjacency = chemical_bond_adjacency(
                input_ids[item_index:item_index + 1]
            )[0]
            one_three = torch.matmul(
                adjacency.float(),
                adjacency.float(),
            ).gt(0)
            intra_excluded = adjacency | one_three
        for start in range(0, length, max(1, residue_chunk_size)):
            stop = min(length, start + max(1, residue_chunk_size))
            left_indices = residue_indices[start:stop]
            residue_greater = (
                residue_indices.view(1, -1) > left_indices.view(-1, 1)
            )
            same_residue = (
                residue_indices.view(1, -1) == left_indices.view(-1, 1)
            )
            if input_ids is None:
                pair_order = residue_greater[:, :, None, None]
            else:
                atom_order = (
                    atom_indices.view(1, 1, -1, 1)
                    < atom_indices.view(1, 1, 1, -1)
                )
                pair_order = (
                    residue_greater[:, :, None, None]
                    | (same_residue[:, :, None, None] & atom_order)
                )
            if not pair_order.any():
                continue
            relative = (
                item_coords[start:stop, None, :, None, :]
                - item_coords[None, :, None, :, :]
            )
            distances = torch.linalg.norm(relative, dim=-1)
            valid = (
                item_mask[start:stop, None, :, None]
                & item_mask[None, :, None, :]
                & pair_order
            )
            adjacent = (
                residue_indices.view(1, -1)
                == left_indices.view(-1, 1) + 1
            )
            covalent = torch.zeros_like(valid)
            covalent[:, :, o3, p] = adjacent
            if intra_excluded is not None:
                for local_index, residue_index in enumerate(
                    range(start, stop)
                ):
                    covalent[
                        local_index,
                        residue_index,
                    ] |= intra_excluded[residue_index]
                c3 = RNA_ATOM_TO_INDEX["C3'"]
                o5 = RNA_ATOM_TO_INDEX["O5'"]
                op1 = RNA_ATOM_TO_INDEX["OP1"]
                op2 = RNA_ATOM_TO_INDEX["OP2"]
                covalent[:, :, c3, p] |= adjacent
                covalent[:, :, o3, o5] |= adjacent
                covalent[:, :, o3, op1] |= adjacent
                covalent[:, :, o3, op2] |= adjacent
            valid &= ~covalent
            threshold = minimum_distance.expand(stop - start, length, -1, -1)
            if valid.any():
                penetration = torch.relu(threshold - distances)
                item_penalty = item_penalty + penetration.square()[valid].sum()
                item_pair_count = item_pair_count + valid.sum()
        if item_pair_count.item() > 0:
            # Normalize violation energy per chemically valid atom. Dividing
            # by every possible nonbonded pair makes a fixed local collision
            # vanish as O(L^-2) in long RNAs; per-atom scaling retains an
            # O(L^-1) local signal while keeping examples equally weighted.
            atom_count = item_mask.sum().clamp(min=1).to(pred.dtype)
            example_losses.append(item_penalty / atom_count)
    if not example_losses:
        return pred.sum() * 0.0
    return torch.stack(example_losses).mean()


# ——— FAPE-inspired local-frame loss ———


def frame_aligned_point_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    coord_mask: torch.Tensor,
    clamp_distance: float = 10.0,
    length_scale: float = 10.0,
    frame_chunk_size: int = 16,
    input_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """All-atom RNA FAPE over every valid residue frame and atom point."""
    if pred.ndim != 4 or target.ndim != 4:
        return pred.sum() * 0.0
    pred_rot, pred_origin, pred_frame_mask = build_residue_frames(
        pred, coord_mask, input_ids
    )
    target_rot, target_origin, target_frame_mask = build_residue_frames(
        target, coord_mask, input_ids
    )
    # Frame supervision is defined by the target. A collapsed prediction has
    # degenerate residue frames, but masking those frames would make FAPE zero
    # and remove every recovery gradient. Use a safe predicted-frame fallback
    # while continuing to score all chemically valid target frames.
    identity = torch.eye(3, dtype=pred.dtype, device=pred.device)
    pred_rot = torch.where(
        pred_frame_mask.unsqueeze(-1).unsqueeze(-1),
        pred_rot,
        identity,
    )
    frame_mask = target_frame_mask
    point_mask = coord_mask.flatten(start_dim=1)
    if not frame_mask.any() or not point_mask.any():
        return pred.sum() * 0.0
    pred_points = pred.flatten(start_dim=1, end_dim=2)
    target_points = target.flatten(start_dim=1, end_dim=2)
    total = torch.zeros(pred.size(0), dtype=pred.dtype, device=pred.device)
    count = torch.zeros(pred.size(0), dtype=torch.long, device=pred.device)
    chunk_size = max(1, int(frame_chunk_size))
    for start in range(0, pred.size(1), chunk_size):
        stop = min(pred.size(1), start + chunk_size)
        pred_delta = (
            pred_points.unsqueeze(1)
            - pred_origin[:, start:stop].unsqueeze(2)
        )
        target_delta = (
            target_points.unsqueeze(1)
            - target_origin[:, start:stop].unsqueeze(2)
        )
        pred_local = torch.einsum(
            "bcji,bcpj->bcpi", pred_rot[:, start:stop], pred_delta
        )
        target_local = torch.einsum(
            "bcji,bcpj->bcpi", target_rot[:, start:stop], target_delta
        )
        error = torch.linalg.norm(pred_local - target_local, dim=-1)
        error = error.clamp(max=clamp_distance) / length_scale
        valid = frame_mask[:, start:stop].unsqueeze(-1) & point_mask.unsqueeze(1)
        total = total + (error * valid).sum(dim=(1, 2))
        count = count + valid.sum(dim=(1, 2))
    valid_examples = count.gt(0)
    if not valid_examples.any():
        return pred.sum() * 0.0
    per_example = total / count.clamp(min=1).to(total.dtype)
    return per_example[valid_examples].mean()


def local_frame_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    coord_mask: torch.Tensor,
    input_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Backward-compatible alias for the full frame-aligned point error."""
    return frame_aligned_point_error(
        pred, target, coord_mask, input_ids=input_ids
    )
