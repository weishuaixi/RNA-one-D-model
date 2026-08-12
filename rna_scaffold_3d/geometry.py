from __future__ import annotations

import torch
import torch.nn.functional as F

from rna_scaffold_3d.rna_atoms import RNA_ATOM_TO_INDEX


def random_rotation_matrix(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample proper 3D rotations from normalized random quaternions."""
    quaternion = torch.randn(
        batch_size,
        4,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    quaternion = F.normalize(quaternion, dim=-1)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).view(batch_size, 3, 3)


def apply_random_rigid_augmentation(
    coords: torch.Tensor,
    coord_mask: torch.Tensor,
    *,
    translation_scale: float = 5.0,
) -> torch.Tensor:
    """Apply one random SE(3) transform per example without moving missing atoms."""
    rotations = random_rotation_matrix(
        coords.size(0),
        device=coords.device,
        dtype=coords.dtype,
    )
    translations = torch.randn(
        coords.size(0),
        3,
        device=coords.device,
        dtype=coords.dtype,
    ) * translation_scale
    transformed = torch.einsum("b...j,bkj->b...k", coords, rotations)
    view_shape = (coords.size(0),) + (1,) * (coords.ndim - 2) + (3,)
    transformed = transformed + translations.view(view_shape)
    return torch.where(coord_mask.unsqueeze(-1), transformed, torch.zeros_like(transformed))


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Continuous 6D rotation representation converted to right-handed matrices."""
    first, second = rotation_6d[..., :3], rotation_6d[..., 3:]
    e1 = F.normalize(first, dim=-1, eps=1e-6)
    second = second - (e1 * second).sum(dim=-1, keepdim=True) * e1
    e2 = F.normalize(second, dim=-1, eps=1e-6)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack((e1, e2, e3), dim=-1)


def build_residue_frames(
    coords: torch.Tensor,
    coord_mask: torch.Tensor,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build RNA residue frames from C4', C1' and glycosidic N1/N9 atoms."""
    c4 = RNA_ATOM_TO_INDEX["C4'"]
    c1 = RNA_ATOM_TO_INDEX["C1'"]
    n1 = RNA_ATOM_TO_INDEX["N1"]
    n9 = RNA_ATOM_TO_INDEX["N9"]
    origins = coords[..., c4, :]
    if input_ids is None:
        use_n9 = coord_mask[..., n9]
        glyco_valid = coord_mask[..., n1] | coord_mask[..., n9]
    else:
        if input_ids.shape != coords.shape[:2]:
            raise ValueError("input_ids must match the batch/residue dimensions.")
        use_n9 = input_ids.eq(1) | input_ids.eq(4)
        use_n1 = input_ids.eq(2) | input_ids.eq(3)
        glyco_valid = (
            (use_n9 & coord_mask[..., n9])
            | (use_n1 & coord_mask[..., n1])
        )
    glyco = torch.where(
        use_n9.unsqueeze(-1),
        coords[..., n9, :],
        coords[..., n1, :],
    )
    valid = (
        coord_mask[..., c4]
        & coord_mask[..., c1]
        & glyco_valid
    )
    c1_direction = coords[..., c1, :] - origins
    glyco_direction = glyco - origins
    normal = torch.cross(c1_direction, glyco_direction, dim=-1)
    nondegenerate = (
        torch.linalg.norm(c1_direction, dim=-1) > 1e-4
    ) & (
        torch.linalg.norm(normal, dim=-1) > 1e-4
    )
    e1 = F.normalize(c1_direction, dim=-1, eps=1e-6)
    e3 = F.normalize(normal, dim=-1, eps=1e-6)
    e2 = torch.cross(e3, e1, dim=-1)
    rotations = torch.stack((e1, e2, e3), dim=-1)
    finite = torch.isfinite(rotations).flatten(start_dim=-2).all(dim=-1)
    return rotations, origins, valid & nondegenerate & finite


def kabsch_align(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Align predicted points to targets independently for every batch item."""
    with torch.autocast(device_type=pred.device.type, enabled=False):
        return _kabsch_align_float(pred.float(), target.float(), mask)


def _kabsch_align_float(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Run the numerically sensitive Kabsch solve outside mixed precision."""
    aligned = pred.clone()
    for batch_index in range(pred.size(0)):
        valid = mask[batch_index]
        if int(valid.sum().item()) < 3:
            continue
        p = pred[batch_index, valid]
        q = target[batch_index, valid]
        p_center = p.mean(dim=0)
        q_center = q.mean(dim=0)
        covariance = (p - p_center).transpose(0, 1) @ (q - q_center)
        # Distinct, scale-aware diagonal jitter keeps SVD gradients finite for
        # collapsed, collinear, or planar point sets while remaining negligible
        # for a well-conditioned covariance.
        jitter = (
            covariance.detach().abs().amax() * 1e-7 + 1e-7
        )
        covariance = covariance + torch.diag(
            covariance.new_tensor((1.0, 2.0, 3.0))
        ) * jitter
        u, _, vh = torch.linalg.svd(covariance, full_matrices=False)
        correction = torch.eye(3, dtype=pred.dtype, device=pred.device)
        correction[-1, -1] = torch.det(vh.transpose(-2, -1) @ u.transpose(-2, -1))
        rotation = vh.transpose(-2, -1) @ correction @ u.transpose(-2, -1)
        aligned[batch_index] = (pred[batch_index] - p_center) @ rotation.transpose(-2, -1) + q_center
    return aligned
