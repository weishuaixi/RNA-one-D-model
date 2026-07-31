from __future__ import annotations

from pathlib import Path

import torch

from rna_scaffold_3d.rna_atoms import RNA_ATOM_NAMES, RNA_ATOM_TO_INDEX, RNA_NUM_ATOMS, atom_names_for_base


def coordinates_to_pdb(
    sequence: str,
    coords: torch.Tensor,
    chain_id: str = "A",
) -> str:
    if coords.ndim != 3:
        raise ValueError("coords must have shape (sequence_length, num_atoms, 3).")
    if coords.shape[0] != len(sequence):
        raise ValueError("coords residue dimension must match sequence length.")
    if coords.shape[1] != RNA_NUM_ATOMS:
        raise ValueError(f"coords must contain {RNA_NUM_ATOMS} canonical RNA heavy atoms per residue.")

    lines: list[str] = []
    serial = 1
    for residue_index, base in enumerate(sequence.upper(), start=1):
        for atom_name in atom_names_for_base(base):
            atom_index = RNA_ATOM_TO_INDEX[atom_name]
            x, y, z = [float(value) for value in coords[residue_index - 1, atom_index].tolist()]
            element = atom_name[0] if atom_name[0].isalpha() else "C"
            lines.append(
                f"ATOM  {serial:5d} {atom_name:>4s}   {base} {chain_id}{residue_index:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {element:>2s}"
            )
            serial += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


def write_pdb(sequence: str, coords: torch.Tensor, path: str | Path, chain_id: str = "A") -> Path:
    pdb_path = Path(path)
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    pdb_path.write_text(coordinates_to_pdb(sequence, coords, chain_id=chain_id), encoding="utf-8")
    return pdb_path


def count_pdb_atoms(pdb_text: str) -> int:
    return sum(1 for line in pdb_text.splitlines() if line.startswith("ATOM"))


def require_complete_pdb(
    pdb_text: str,
    sequence_length: int | None = None,
    sequence: str | None = None,
) -> None:
    if sequence is not None:
        expected_atoms = sum(len(atom_names_for_base(base)) for base in sequence)
        sequence_length = len(sequence)
    elif sequence_length is not None:
        # Legacy validation cannot know base-specific atom presence.
        expected_atoms = sequence_length * RNA_NUM_ATOMS
    else:
        raise ValueError("sequence or sequence_length is required.")
    actual_atoms = count_pdb_atoms(pdb_text)
    if actual_atoms != expected_atoms:
        raise ValueError(
            f"Incomplete RNA PDB: expected {expected_atoms} canonical heavy atoms "
            f"for {sequence_length} residues, found {actual_atoms}."
        )
