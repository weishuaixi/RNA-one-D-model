from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from rna_scaffold_3d.rna_atoms import RNA_ATOM_TO_INDEX


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


def masked_coordinate_mse(pred: torch.Tensor, target: torch.Tensor, coord_mask: torch.Tensor) -> torch.Tensor:
    if pred.ndim == 4 and target.ndim == 3:
        pred = pred[..., 0, :]
    if not coord_mask.any():
        return pred.sum() * 0.0
    diff = pred[coord_mask] - target[coord_mask]
    return diff.pow(2).mean()


def masked_coordinate_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    coord_mask: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    if pred.ndim == 4 and target.ndim == 3:
        pred = pred[..., 0, :]
    if not coord_mask.any():
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred[coord_mask], target[coord_mask], beta=beta)


def masked_pairwise_distance_mse(pred: torch.Tensor, target: torch.Tensor, coord_mask: torch.Tensor) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    if pred.ndim == 4:
        pred = pred[..., 0, :]
        if target.ndim == 4:
            target = target[..., 0, :]
            coord_mask = coord_mask[..., 0]
    for pred_item, target_item, mask_item in zip(pred, target, coord_mask):
        if int(mask_item.sum().item()) < 2:
            continue
        pred_valid = pred_item[mask_item]
        target_valid = target_item[mask_item]
        pred_dist = torch.cdist(pred_valid, pred_valid)
        target_dist = torch.cdist(target_valid, target_valid)
        losses.append((pred_dist - target_dist).pow(2).mean())
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def pair_distance_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    coord_mask: torch.Tensor,
    bin_edges: torch.Tensor | None = None,
    atom_index: int = 0,
) -> torch.Tensor:
    if bin_edges is None:
        bin_edges = torch.linspace(2.0, 40.0, logits.size(-1) - 1, device=logits.device)
    else:
        bin_edges = bin_edges.to(logits.device)
    if target.ndim == 4:
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
    ("C5'", "C4'", "C3'", 110.0),
    ("C4'", "C3'", "O3'", 110.0),
)

_TORSION_SPECS = (
    ("P", "O5'", "C5'", "C4'"),
    ("O5'", "C5'", "C4'", "C3'"),
    ("C5'", "C4'", "C3'", "O3'"),
)


def bond_length_loss(pred: torch.Tensor, coord_mask: torch.Tensor) -> torch.Tensor:
    if pred.ndim != 4:
        return pred.sum() * 0.0
    losses: list[torch.Tensor] = []
    for atom_a, atom_b, target_length in _BOND_SPECS:
        idx_a = RNA_ATOM_TO_INDEX[atom_a]
        idx_b = RNA_ATOM_TO_INDEX[atom_b]
        valid = coord_mask[..., idx_a] & coord_mask[..., idx_b]
        if valid.any():
            distances = torch.linalg.norm(pred[..., idx_a, :] - pred[..., idx_b, :], dim=-1)
            losses.append((distances[valid] - target_length).pow(2).mean())
    return torch.stack(losses).mean() if losses else pred.sum() * 0.0


def bond_angle_loss(pred: torch.Tensor, coord_mask: torch.Tensor) -> torch.Tensor:
    if pred.ndim != 4:
        return pred.sum() * 0.0
    losses: list[torch.Tensor] = []
    for atom_a, atom_b, atom_c, target_degrees in _ANGLE_SPECS:
        idx_a = RNA_ATOM_TO_INDEX[atom_a]
        idx_b = RNA_ATOM_TO_INDEX[atom_b]
        idx_c = RNA_ATOM_TO_INDEX[atom_c]
        valid = coord_mask[..., idx_a] & coord_mask[..., idx_b] & coord_mask[..., idx_c]
        if valid.any():
            ba = pred[..., idx_a, :] - pred[..., idx_b, :]
            bc = pred[..., idx_c, :] - pred[..., idx_b, :]
            cos_angle = F.cosine_similarity(ba, bc, dim=-1).clamp(-1.0, 1.0)
            angle = torch.rad2deg(torch.acos(cos_angle))
            losses.append((angle[valid] - target_degrees).pow(2).mean() / 100.0)
    return torch.stack(losses).mean() if losses else pred.sum() * 0.0


def torsion_angle_loss(pred: torch.Tensor, coord_mask: torch.Tensor) -> torch.Tensor:
    if pred.ndim != 4:
        return pred.sum() * 0.0
    losses: list[torch.Tensor] = []
    for atom_a, atom_b, atom_c, atom_d in _TORSION_SPECS:
        indices = [RNA_ATOM_TO_INDEX[a] for a in (atom_a, atom_b, atom_c, atom_d)]
        valid = coord_mask[..., indices[0]] & coord_mask[..., indices[1]] & coord_mask[..., indices[2]] & coord_mask[..., indices[3]]
        if valid.any():
            angle = _dihedral(pred[..., indices[0], :], pred[..., indices[1], :], pred[..., indices[2], :], pred[..., indices[3], :])
            # Mild regularizer: avoid collapsed undefined torsions by favoring finite smooth angles.
            losses.append((1.0 - torch.cos(angle[valid])).mean())
    return torch.stack(losses).mean() if losses else pred.sum() * 0.0


def plddt_confidence_loss(
    predicted_plddt: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    coord_mask: torch.Tensor,
    distance_scale: float = 4.0,
) -> torch.Tensor:
    if pred.ndim != 4 or target.ndim != 4:
        return predicted_plddt.sum() * 0.0
    residue_mask = coord_mask.any(dim=-1)
    if not residue_mask.any():
        return predicted_plddt.sum() * 0.0
    atom_counts = coord_mask.sum(dim=-1).clamp(min=1).to(pred.dtype)
    residue_rmse = ((pred - target).pow(2).sum(dim=-1) * coord_mask.to(pred.dtype)).sum(dim=-1)
    residue_rmse = torch.sqrt(residue_rmse / atom_counts + 1e-8)
    target_confidence = 100.0 * torch.exp(-residue_rmse.detach() / distance_scale)
    return F.smooth_l1_loss(predicted_plddt[residue_mask], target_confidence[residue_mask], beta=5.0)


def _dihedral(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = F.normalize(b1, dim=-1)
    v = b0 - (b0 * b1).sum(dim=-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(dim=-1, keepdim=True) * b1
    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1, v, dim=-1) * w).sum(dim=-1)
    return torch.atan2(y, x)


# ——— steric clash penalty ———


def steric_clash_loss(
    pred: torch.Tensor,
    coord_mask: torch.Tensor,
    clash_threshold: float = 2.0,
) -> torch.Tensor:
    """Penalize non-bonded atoms that are unrealistically close.

    For each RNA in the batch, compute all pairwise atom distances
    between residues |i-j| ≥ 2, and softly penalize pairs closer than
    *clash_threshold* (Å).
    """
    if pred.ndim != 4:
        return pred.sum() * 0.0

    device = pred.device
    vdw = _ATOM_VDW.to(device)
    total_penalty = torch.tensor(0.0, device=device)
    count = 0

    for item_coords, item_mask in zip(pred, coord_mask):
        n_res, n_atoms = item_coords.shape[:2]
        if n_res < 3:
            continue

        # valid residues: at least one atom present
        res_valid = item_mask.any(dim=1)
        valid_indices = res_valid.nonzero(as_tuple=True)[0]
        if len(valid_indices) < 3:
            continue

        # Compute residue-level representative (C4' atom at index 5)
        c4_idx = 5
        reps = item_coords[:, c4_idx, :]  # (n_res, 3)
        rep_mask = item_mask[:, c4_idx]  # (n_res,)

        # Only consider valid residues with C4' present
        valid_with_c4 = valid_indices[rep_mask[valid_indices]]
        if len(valid_with_c4) < 3:
            continue

        reps_valid = reps[valid_with_c4]
        dists = torch.cdist(reps_valid, reps_valid)  # (n_valid, n_valid)

        # Penalize residue pairs with |i-j| >= 2 and distance < clash_threshold
        idx_diff = (valid_with_c4.unsqueeze(1) - valid_with_c4.unsqueeze(0)).abs()
        clash_mask = (idx_diff >= 2) & (dists < clash_threshold) & (dists > 0)
        if clash_mask.any():
            total_penalty = total_penalty + (clash_threshold - dists[clash_mask]).pow(2).mean()
            count += 1

    if count == 0:
        return pred.sum() * 0.0
    return total_penalty / count


# ——— FAPE-inspired local-frame loss ———


def local_frame_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    coord_mask: torch.Tensor,
) -> torch.Tensor:
    """Simplified FAPE: compute per-residue orthonormal frames from backbone
    atoms, transform coordinates into each local frame, then compute MSE.

    Each residue frame is built from:
      - origin  = C4'  (atom index 5)
      - e1       = normalize(C4' → C1')  (C1' = atom index 11)
      - e3       = normalize(e1 × (C4' → N1/N9))
      - e2       = e3 × e1

    For purines N9 (index 22), for pyrimidines N1 (index 12).
    """
    if pred.ndim != 4:
        return pred.sum() * 0.0

    device = pred.device
    c4_idx, c1_idx, n1_idx, n9_idx = 5, 11, 12, 22

    def build_frames(coords, mask):
        """Return (n_valid, 3, 3) rotation matrices and (n_valid, 3) origins."""
        B, R = coords.shape[:2]
        origins = coords[:, :, c4_idx, :]  # (B, R, 3)
        c4_mask = mask[:, :, c4_idx]
        c1_coords = coords[:, :, c1_idx, :]
        c1_mask = mask[:, :, c1_idx]

        # Choose N1 or N9 based on which is present
        n1_coords = coords[:, :, n1_idx, :]
        n1_m = mask[:, :, n1_idx]
        n9_coords = coords[:, :, n9_idx, :]
        n9_m = mask[:, :, n9_idx]
        n_coords = torch.where(n1_m.unsqueeze(-1), n1_coords, n9_coords)
        n_mask = n1_m | n9_m

        valid = c4_mask & c1_mask & n_mask

        # e1 = normalize(C1' - C4')
        e1 = nn.functional.normalize(c1_coords - origins, dim=-1)

        # e3 = normalize(e1 × (n - C4'))
        n_vec = n_coords - origins
        e3 = nn.functional.normalize(torch.cross(e1, n_vec, dim=-1), dim=-1)

        # e2 = e3 × e1
        e2 = torch.cross(e3, e1, dim=-1)

        return origins, e1, e2, e3, valid

    pred_orig, pred_e1, pred_e2, pred_e3, pred_valid = build_frames(pred, coord_mask)
    tgt_orig, tgt_e1, tgt_e2, tgt_e3, tgt_valid = build_frames(target, coord_mask)
    valid = pred_valid & tgt_valid

    if valid.sum() < 2:
        return pred.sum() * 0.0

    total = torch.tensor(0.0, device=device)
    count = 0

    # For each valid residue, transform atom coordinates to its local frame
    # and compare predicted vs target
    for b in range(pred.shape[0]):
        batch_valid = valid[b].nonzero(as_tuple=True)[0]
        if len(batch_valid) < 1:
            continue
        for res_i in batch_valid.tolist():
            # Build rotation matrix for this residue
            R_pred = torch.stack([
                pred_e1[b, res_i],
                pred_e2[b, res_i],
                pred_e3[b, res_i],
            ], dim=0)  # (3, 3)

            R_tgt = torch.stack([
                tgt_e1[b, res_i],
                tgt_e2[b, res_i],
                tgt_e3[b, res_i],
            ], dim=0)

            # Transform atoms to local frame
            atom_mask = coord_mask[b, res_i]  # (n_atoms,)
            if atom_mask.sum() < 1:
                continue
            pred_local = (pred[b, res_i, atom_mask] - pred_orig[b, res_i]) @ R_pred.T
            tgt_local = (target[b, res_i, atom_mask] - tgt_orig[b, res_i]) @ R_tgt.T

            total = total + (pred_local - tgt_local).pow(2).mean()
            count += 1

    if count == 0:
        return pred.sum() * 0.0
    return total / count
