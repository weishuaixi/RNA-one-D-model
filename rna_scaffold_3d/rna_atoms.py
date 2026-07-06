from __future__ import annotations


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
