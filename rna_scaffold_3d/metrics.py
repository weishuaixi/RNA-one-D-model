from __future__ import annotations

import torch

from rna_scaffold_3d.geometry import kabsch_align
from rna_scaffold_3d.rna_atoms import RNA_ATOM_TO_INDEX


def representative_points(
    pred: torch.Tensor,
    target: torch.Tensor,
    coord_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return predicted/target C1' points and a residue-valid mask."""
    c1 = RNA_ATOM_TO_INDEX["C1'"]
    pred_points = pred[..., c1, :] if pred.ndim == 4 else pred
    if target.ndim == 4:
        return pred_points, target[..., c1, :], coord_mask[..., c1]
    return pred_points, target, coord_mask


def batch_structure_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    coord_mask: torch.Tensor,
    *,
    lddt_cutoff: float = 15.0,
) -> dict[str, torch.Tensor]:
    """Compute rigid-invariant C1' metrics, averaging valid examples in a batch."""
    pred_points, target_points, point_mask = representative_points(pred, target, coord_mask)
    aligned = kabsch_align(pred_points.float(), target_points.float(), point_mask)
    rmsd_values: list[torch.Tensor] = []
    drmsd_values: list[torch.Tensor] = []
    lddt_values: list[torch.Tensor] = []
    adjacent_values: list[torch.Tensor] = []
    for predicted, aligned_predicted, observed, valid in zip(
        pred_points.float(), aligned, target_points.float(), point_mask
    ):
        if valid.any():
            rmsd_values.append(
                torch.sqrt((aligned_predicted[valid] - observed[valid]).pow(2).sum(-1).mean())
            )
        if int(valid.sum().item()) >= 2:
            predicted_valid = predicted[valid]
            observed_valid = observed[valid]
            pred_distance = torch.cdist(predicted_valid, predicted_valid)
            target_distance = torch.cdist(observed_valid, observed_valid)
            drmsd_values.append(torch.sqrt((pred_distance - target_distance).pow(2).mean()))
            identity = torch.eye(predicted_valid.size(0), dtype=torch.bool, device=pred.device)
            neighborhood = ~identity & target_distance.lt(lddt_cutoff)
            if neighborhood.any():
                error = (pred_distance - target_distance).abs()
                score = torch.stack(
                    [(error < threshold).float() for threshold in (0.5, 1.0, 2.0, 4.0)]
                ).mean(0)
                lddt_values.append(score[neighborhood].mean() * 100.0)
        adjacent_mask = valid[:-1] & valid[1:]
        if adjacent_mask.any():
            distance = torch.linalg.norm(predicted[1:] - predicted[:-1], dim=-1)
            adjacent_values.append(distance[adjacent_mask].mean())

    zero = pred.sum().detach() * 0.0

    def average(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values).mean() if values else zero

    return {
        "kabsch_rmsd": average(rmsd_values),
        "kabsch_rmsd_count": torch.tensor(
            len(rmsd_values), device=pred.device, dtype=torch.float32
        ),
        "distance_rmsd": average(drmsd_values),
        "distance_rmsd_count": torch.tensor(
            len(drmsd_values), device=pred.device, dtype=torch.float32
        ),
        "c1_lddt": average(lddt_values),
        "c1_lddt_count": torch.tensor(
            len(lddt_values), device=pred.device, dtype=torch.float32
        ),
        "adjacent_c1_mean": average(adjacent_values),
        "adjacent_c1_mean_count": torch.tensor(
            len(adjacent_values), device=pred.device, dtype=torch.float32
        ),
    }
