from __future__ import annotations

from pathlib import Path

import torch


def coordinates_to_pdb(sequence: str, coords: torch.Tensor, chain_id: str = "A") -> str:
    lines: list[str] = []
    for index, (base, coord) in enumerate(zip(sequence.upper(), coords), start=1):
        x, y, z = [float(value) for value in coord.tolist()]
        lines.append(
            f"ATOM  {index:5d}  C4'   {base} {chain_id}{index:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    return "\n".join(lines) + "\n"


def write_pdb(sequence: str, coords: torch.Tensor, path: str | Path, chain_id: str = "A") -> Path:
    pdb_path = Path(path)
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    pdb_path.write_text(coordinates_to_pdb(sequence, coords, chain_id=chain_id), encoding="utf-8")
    return pdb_path
