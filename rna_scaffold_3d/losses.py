from __future__ import annotations

import torch


def masked_coordinate_mse(pred: torch.Tensor, target: torch.Tensor, coord_mask: torch.Tensor) -> torch.Tensor:
    if not coord_mask.any():
        return pred.sum() * 0.0
    diff = pred[coord_mask] - target[coord_mask]
    return diff.pow(2).mean()


def masked_pairwise_distance_mse(pred: torch.Tensor, target: torch.Tensor, coord_mask: torch.Tensor) -> torch.Tensor:
    losses: list[torch.Tensor] = []
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
