from __future__ import annotations

import torch


RNA_ATOM_NAMES = (
    "P",
    "OP1",
    "OP2",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
    "N1",
    "C2",
    "O2",
    "N2",
    "N3",
    "C4",
    "N4",
    "C5",
    "C6",
    "O4",
    "N9",
    "C8",
    "N7",
    "N6",
    "O6",
)

RNA_ATOM_TO_INDEX = {atom: index for index, atom in enumerate(RNA_ATOM_NAMES)}
RNA_NUM_ATOMS = len(RNA_ATOM_NAMES)

RNA_BACKBONE_ATOMS = RNA_ATOM_NAMES[:12]
RNA_BASE_ATOMS = {
    "A": ("N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"),
    "G": ("N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"),
    "C": ("N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"),
    "U": ("N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"),
}

RNA_COMMON_BOND_LENGTHS = (
    ("P", "OP1", 1.48),
    ("P", "OP2", 1.48),
    ("P", "O5'", 1.60),
    ("O5'", "C5'", 1.43),
    ("C5'", "C4'", 1.52),
    ("C4'", "O4'", 1.45),
    ("C4'", "C3'", 1.53),
    ("O4'", "C1'", 1.44),
    ("C3'", "O3'", 1.42),
    ("C3'", "C2'", 1.53),
    ("C2'", "O2'", 1.43),
    ("C2'", "C1'", 1.54),
)
RNA_BASE_BOND_LENGTHS = {
    "A": (
        ("C1'", "N9", 1.464), ("N9", "C8", 1.363), ("N9", "C4", 1.372),
        ("C8", "N7", 1.301), ("N7", "C5", 1.354), ("C5", "C6", 1.405),
        ("C5", "C4", 1.405), ("C6", "N6", 1.384), ("C6", "N1", 1.328),
        ("N1", "C2", 1.320), ("C2", "N3", 1.316), ("N3", "C4", 1.328),
    ),
    "U": (
        ("C1'", "N1", 1.466), ("N1", "C2", 1.344), ("N1", "C6", 1.368),
        ("C2", "O2", 1.215), ("C2", "N3", 1.347), ("N3", "C4", 1.348),
        ("C4", "O4", 1.218), ("C4", "C5", 1.415), ("C5", "C6", 1.350),
    ),
    "C": (
        ("C1'", "N1", 1.465), ("N1", "C2", 1.344), ("N1", "C6", 1.362),
        ("C2", "O2", 1.220), ("C2", "N3", 1.332), ("N3", "C4", 1.326),
        ("C4", "N4", 1.375), ("C4", "C5", 1.410), ("C5", "C6", 1.354),
    ),
    "G": (
        ("C1'", "N9", 1.465), ("N9", "C8", 1.365), ("N9", "C4", 1.368),
        ("C8", "N7", 1.301), ("N7", "C5", 1.356), ("C5", "C6", 1.414),
        ("C5", "C4", 1.399), ("C6", "O6", 1.219), ("C6", "N1", 1.351),
        ("N1", "C2", 1.362), ("C2", "N2", 1.372), ("C2", "N3", 1.314),
        ("N3", "C4", 1.338),
    ),
}


def atom_names_for_base(base: str) -> tuple[str, ...]:
    normalized = base.upper().replace("T", "U")
    if normalized not in RNA_BASE_ATOMS:
        raise ValueError(f"Unsupported RNA base: {base!r}")
    return RNA_BACKBONE_ATOMS + RNA_BASE_ATOMS[normalized]


def chemical_atom_mask(input_ids: torch.Tensor) -> torch.Tensor:
    """Return chemically present heavy-atom slots for A/U/C/G token IDs.

    Token IDs follow the 3D vocabulary: PAD=0, A=1, U=2, C=3, G=4,
    MASK=5. PAD/MASK/unknown IDs intentionally contain no atoms; training
    should call this with the unmasked target sequence.
    """
    lookup = torch.zeros(
        6,
        RNA_NUM_ATOMS,
        dtype=torch.bool,
        device=input_ids.device,
    )
    for token_id, base in enumerate(("A", "U", "C", "G"), start=1):
        indices = [
            RNA_ATOM_TO_INDEX[atom_name]
            for atom_name in atom_names_for_base(base)
        ]
        lookup[token_id, indices] = True
    safe_ids = input_ids.clamp(min=0, max=lookup.size(0) - 1)
    mask = lookup[safe_ids]
    known = input_ids.ge(0) & input_ids.lt(lookup.size(0))
    return mask & known.unsqueeze(-1)


def chemical_bond_adjacency(input_ids: torch.Tensor) -> torch.Tensor:
    """Return symmetric within-residue 1–2 covalent adjacency."""
    lookup = torch.zeros(
        6,
        RNA_NUM_ATOMS,
        RNA_NUM_ATOMS,
        dtype=torch.bool,
        device=input_ids.device,
    )
    for token_id, base in enumerate(("A", "U", "C", "G"), start=1):
        for atom_a, atom_b, _ in (
            RNA_COMMON_BOND_LENGTHS + RNA_BASE_BOND_LENGTHS[base]
        ):
            index_a = RNA_ATOM_TO_INDEX[atom_a]
            index_b = RNA_ATOM_TO_INDEX[atom_b]
            lookup[token_id, index_a, index_b] = True
            lookup[token_id, index_b, index_a] = True
    safe_ids = input_ids.clamp(min=0, max=lookup.size(0) - 1)
    adjacency = lookup[safe_ids]
    known = input_ids.ge(1) & input_ids.le(4)
    return adjacency & known.unsqueeze(-1).unsqueeze(-1)
