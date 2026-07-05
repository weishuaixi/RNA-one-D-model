from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rna_scaffold_3d.data import RNA_BASE_TO_ID
from rna_scaffold_3d.model import Rna3DCoordinatePredictor
from rna_scaffold_3d.pdb_writer import write_pdb
from rna_scaffold.utils import validate_rna_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict pseudo C4' RNA 3D coordinates from a trained local checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output-pdb", default="outputs/rna_3d/predicted.pdb")
    args = parser.parse_args()

    sequence = args.sequence.strip().upper().replace("T", "U")
    if not validate_rna_sequence(sequence):
        raise ValueError("sequence must contain only A, U, C, and G.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = Rna3DCoordinatePredictor(**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    input_ids = torch.tensor([[RNA_BASE_TO_ID[base] for base in sequence]], dtype=torch.long)
    padding_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    with torch.inference_mode():
        coords = model(input_ids=input_ids, padding_mask=padding_mask)[0]
    pdb_path = write_pdb(sequence=sequence, coords=coords, path=Path(args.output_pdb))
    print(str(pdb_path))


if __name__ == "__main__":
    main()
