from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from rna_scaffold_3d.rna_atoms import RNA_ATOM_TO_INDEX, RNA_NUM_ATOMS


# Bond lengths (Å) and valence angles (degrees) for the covalent RNA backbone.
_BACKBONE_LENGTHS = {
    "P_O5": 1.60,
    "O5_C5": 1.43,
    "C5_C4": 1.52,
    "C4_C3": 1.53,
    "C3_O3": 1.42,
    "O3_P": 1.60,
}
_BACKBONE_ANGLES = {
    "O3_P_O5": 104.0,
    "P_O5_C5": 120.0,
    "O5_C5_C4": 111.0,
    "C5_C4_C3": 116.0,
    "C4_C3_O3": 113.0,
    "C3_O3_P": 120.0,
}


def place_atom(
    atom_a: torch.Tensor,
    atom_b: torch.Tensor,
    atom_c: torch.Tensor,
    bond_length: float,
    bond_angle_degrees: float,
    dihedral: torch.Tensor,
) -> torch.Tensor:
    """Place D from A-B-C using a differentiable NeRF internal-coordinate step."""
    bc = F.normalize(atom_c - atom_b, dim=-1, eps=1e-7)
    plane_normal = F.normalize(
        torch.cross(atom_b - atom_a, bc, dim=-1), dim=-1, eps=1e-7
    )
    in_plane = torch.cross(plane_normal, bc, dim=-1)
    angle = math.radians(bond_angle_degrees)
    direction = (
        -math.cos(angle) * bc
        + math.sin(angle) * torch.cos(dihedral).unsqueeze(-1) * in_plane
        + math.sin(angle) * torch.sin(dihedral).unsqueeze(-1) * plane_normal
    )
    return atom_c + bond_length * direction


def _dihedral(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
) -> torch.Tensor:
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = F.normalize(b1, dim=-1, eps=1e-7)
    v = b0 - (b0 * b1).sum(dim=-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(dim=-1, keepdim=True) * b1
    y = (torch.cross(b1, v, dim=-1) * w).sum(dim=-1)
    x = (v * w).sum(dim=-1)
    degenerate = (x.square() + y.square()).detach().lt(1e-12)
    safe_x = torch.where(degenerate, torch.ones_like(x), x)
    safe_y = torch.where(degenerate, torch.zeros_like(y), y)
    return torch.atan2(safe_y, safe_x)


def _residue_frames(
    c4: torch.Tensor,
    c3: torch.Tensor,
    c5: torch.Tensor,
) -> torch.Tensor:
    e1 = F.normalize(c3 - c4, dim=-1, eps=1e-7)
    c5_direction = c5 - c4
    c5_orthogonal = c5_direction - (c5_direction * e1).sum(dim=-1, keepdim=True) * e1
    e2 = F.normalize(c5_orthogonal, dim=-1, eps=1e-7)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack((e1, e2, e3), dim=-1)


def build_rna_backbone(
    torsions: torch.Tensor,
    padding_mask: torch.Tensor,
    initial_rotation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build P/O5'/C5'/C4'/C3'/O3' from alpha..zeta torsions."""
    if torsions.ndim != 4 or torsions.size(-2) != 7 or torsions.size(-1) != 2:
        raise ValueError("torsions must have shape [batch, length, 7, 2].")
    batch, length = torsions.shape[:2]
    angles = torch.atan2(torsions[..., 0], torsions[..., 1])
    dtype, device = torsions.dtype, torsions.device
    previous_o3 = torch.tensor([-1.0, -1.25, 0.0], dtype=dtype, device=device).expand(batch, 3)
    p = torch.zeros(batch, 3, dtype=dtype, device=device)
    o5 = torch.tensor([_BACKBONE_LENGTHS["P_O5"], 0.0, 0.0], dtype=dtype, device=device).expand(batch, 3)

    residues: list[torch.Tensor] = []
    frames: list[torch.Tensor] = []
    origins: list[torch.Tensor] = []
    for index in range(length):
        alpha, beta, gamma, delta, epsilon, zeta = angles[:, index, :6].unbind(dim=-1)
        c5 = place_atom(
            previous_o3, p, o5, _BACKBONE_LENGTHS["O5_C5"],
            _BACKBONE_ANGLES["P_O5_C5"], alpha,
        )
        c4 = place_atom(
            p, o5, c5, _BACKBONE_LENGTHS["C5_C4"],
            _BACKBONE_ANGLES["O5_C5_C4"], beta,
        )
        c3 = place_atom(
            o5, c5, c4, _BACKBONE_LENGTHS["C4_C3"],
            _BACKBONE_ANGLES["C5_C4_C3"], gamma,
        )
        o3 = place_atom(
            c5, c4, c3, _BACKBONE_LENGTHS["C3_O3"],
            _BACKBONE_ANGLES["C4_C3_O3"], delta,
        )
        residue = torch.stack((p, o5, c5, c4, c3, o3), dim=-2)
        residues.append(residue)
        frames.append(_residue_frames(c4, c3, c5))
        origins.append(c4)
        if index + 1 < length:
            next_p = place_atom(
                c4, c3, o3, _BACKBONE_LENGTHS["O3_P"],
                _BACKBONE_ANGLES["C3_O3_P"], epsilon,
            )
            next_o5 = place_atom(
                c3, o3, next_p, _BACKBONE_LENGTHS["P_O5"],
                _BACKBONE_ANGLES["O3_P_O5"], zeta,
            )
            previous_o3, p, o5 = o3, next_p, next_o5

    backbone = torch.stack(residues, dim=1)
    residue_frames = torch.stack(frames, dim=1)
    residue_origins = torch.stack(origins, dim=1)
    backbone = torch.einsum("bij,blaj->blai", initial_rotation, backbone)
    residue_origins = torch.einsum("bij,blj->bli", initial_rotation, residue_origins)
    residue_frames = torch.einsum("bij,bljk->blik", initial_rotation, residue_frames)
    valid = ~padding_mask
    backbone = backbone.masked_fill(~valid[:, :, None, None], 0.0)
    residue_origins = residue_origins.masked_fill(~valid.unsqueeze(-1), 0.0)
    residue_frames = residue_frames.masked_fill(~valid[:, :, None, None], 0.0)
    return backbone, residue_frames, residue_origins


def build_all_atom_from_internal(
    torsions: torch.Tensor,
    sugar_pucker: torch.Tensor,
    base_orientation: torch.Tensor,
    padding_mask: torch.Tensor,
    initial_rotation: torch.Tensor,
    atom_template: torch.Tensor,
    glycosidic_vector: torch.Tensor,
    chi_reference_vector: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct full atom slots from backbone, pucker, chi and base orientation."""
    backbone, frames, origins = build_rna_backbone(
        torsions, padding_mask, initial_rotation
    )
    batch, length = torsions.shape[:2]
    coords = torch.zeros(
        batch, length, RNA_NUM_ATOMS, 3,
        dtype=torsions.dtype, device=torsions.device,
    )
    for backbone_index, atom_name in enumerate(("P", "O5'", "C5'", "C4'", "C3'", "O3'")):
        coords[..., RNA_ATOM_TO_INDEX[atom_name], :] = backbone[..., backbone_index, :]

    # Canonical C3'-endo ribose geometry from the mean of the wwPDB CCD
    # ideal A/U/C/G components in the C4'->C3'/C5' residue frame.
    pucker_phase = torch.atan2(sugar_pucker[..., 0], sugar_pucker[..., 1])
    reference_phase = math.radians(18.0)
    ring_local = {
        # C3'-endo reference coordinates in the same C4'->C3'/C5'
        # frame used by the builder. An observed C3'-endo residue was
        # projected onto the standard ring bond lengths and an 18-degree
        # pseudorotation phase; averaging Cartesian coordinates would not
        # preserve these nonlinear constraints.
        "O4'": (-0.3444, -0.7248, -1.2077, 0),
        "C1'": (0.7418, -1.6021, -1.5598, 1),
        "C2'": (1.8334, -1.4176, -0.4893, 2),
    }
    for atom_name, (local_x, local_y, local_z, ring_index) in ring_local.items():
        atom_index = RNA_ATOM_TO_INDEX[atom_name]
        local = torch.zeros(batch, length, 3, dtype=torsions.dtype, device=torsions.device)
        local[..., 0] = local_x
        local[..., 1] = local_y
        local[..., 2] = local_z + 0.18 * (
            torch.cos(
                pucker_phase + ring_index * 4.0 * math.pi / 5.0
            )
            - math.cos(reference_phase + ring_index * 4.0 * math.pi / 5.0)
        )
        coords[..., atom_index, :] = (
            origins + torch.einsum("blij,blj->bli", frames, local)
        )
    c2 = coords[..., RNA_ATOM_TO_INDEX["C2'"], :]
    canonical_o2_minus_c2 = torch.tensor(
        [0.7486, -0.9687, 0.7372],
        dtype=torsions.dtype,
        device=torsions.device,
    )
    coords[..., RNA_ATOM_TO_INDEX["O2'"], :] = c2 + torch.einsum(
        "blij,j->bli", frames, canonical_o2_minus_c2
    )

    # Place phosphate oxygens in the tetrahedral P/O5'/previous-O3' frame.
    # Local vectors are the mean wwPDB CCD A/U/C/G ideal coordinates.
    p = coords[..., RNA_ATOM_TO_INDEX["P"], :]
    o5 = coords[..., RNA_ATOM_TO_INDEX["O5'"], :]
    p_axis = F.normalize(o5 - p, dim=-1, eps=1e-7)
    previous_o3 = torch.cat(
        (
            p[:, :1] + frames[:, :1, :, 1],
            coords[:, :-1, RNA_ATOM_TO_INDEX["O3'"], :],
        ),
        dim=1,
    )
    previous_direction = previous_o3 - p
    phosphate_side = previous_direction - (
        previous_direction * p_axis
    ).sum(dim=-1, keepdim=True) * p_axis
    phosphate_side = F.normalize(phosphate_side, dim=-1, eps=1e-7)
    phosphate_normal = torch.cross(p_axis, phosphate_side, dim=-1)
    coords[..., RNA_ATOM_TO_INDEX["OP1"], :] = p + (
        -0.4931 * p_axis
        - 0.6981 * phosphate_side
        + 1.2071 * phosphate_normal
    )
    coords[..., RNA_ATOM_TO_INDEX["OP2"], :] = p + (
        -0.5364 * p_axis
        - 0.7576 * phosphate_side
        - 1.3152 * phosphate_normal
    )

    c1_index = RNA_ATOM_TO_INDEX["C1'"]
    c1 = coords[..., c1_index, :]
    if atom_template.ndim == 2:
        atom_template = atom_template.view(1, 1, *atom_template.shape).expand(
            batch, length, -1, -1
        )
    if atom_template.shape[:2] != (batch, length):
        raise ValueError(
            "atom_template must be [atoms, 3] or [batch, length, atoms, 3]."
        )
    template_c1 = atom_template[..., c1_index, :]
    template_vectors = atom_template - template_c1.unsqueeze(-2)
    oriented_vectors = torch.einsum(
        "blij,blaj->blai", base_orientation, template_vectors
    )
    oriented_glycosidic = torch.einsum(
        "blij,blj->bli", base_orientation, glycosidic_vector
    )
    oriented_reference = torch.einsum(
        "blij,blj->bli", base_orientation, chi_reference_vector
    )
    glycosidic_global = c1 + torch.einsum(
        "blij,blj->bli", frames, oriented_glycosidic
    )
    reference_global = c1 + torch.einsum(
        "blij,blj->bli", frames, oriented_reference
    )
    o4 = coords[..., RNA_ATOM_TO_INDEX["O4'"], :]
    reference_chi = _dihedral(o4, c1, glycosidic_global, reference_global)
    target_chi = torch.atan2(torsions[..., 6, 0], torsions[..., 6, 1])
    chi_delta = target_chi - reference_chi
    axis = F.normalize(oriented_glycosidic, dim=-1, eps=1e-7)
    sin_delta = torch.sin(chi_delta).unsqueeze(-1).unsqueeze(-1)
    cos_delta = torch.cos(chi_delta).unsqueeze(-1).unsqueeze(-1)
    axis_expanded = axis.unsqueeze(-2)
    rotated_vectors = (
        oriented_vectors * cos_delta
        + torch.cross(
            axis_expanded.expand_as(oriented_vectors),
            oriented_vectors,
            dim=-1,
        ) * sin_delta
        + axis_expanded
        * (axis_expanded * oriented_vectors).sum(dim=-1, keepdim=True)
        * (1.0 - cos_delta)
    )
    base_atom_names = (
        "N1", "C2", "O2", "N2", "N3", "C4", "N4", "C5",
        "C6", "O4", "N9", "C8", "N7", "N6", "O6",
    )
    for atom_name in base_atom_names:
        atom_index = RNA_ATOM_TO_INDEX[atom_name]
        local = rotated_vectors[..., atom_index, :]
        coords[..., atom_index, :] = c1 + torch.einsum(
            "blij,blj->bli", frames, local.to(torsions.dtype)
        )

    coords = coords.masked_fill(padding_mask[:, :, None, None], 0.0)
    return coords, frames, origins
