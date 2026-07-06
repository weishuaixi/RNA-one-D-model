from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from rna_scaffold.generate import generate_rna_sequence
from rna_scaffold.utils import validate_rna_sequence
from rna_scaffold_3d.rhofold import RhoFoldConfig, RhoFoldModel
from rna_scaffold_3d.pdb_writer import write_pdb
from rna_scaffold_3d.sequence import encode_rna_sequence


@dataclass(frozen=True)
class FoldResult:
    sequence: str
    coords: torch.Tensor
    plddt: torch.Tensor


def generate_sequence_from_motif(
    motif: str,
    num_candidates: int = 128,
    min_total_length: int | None = None,
    max_total_length: int | None = None,
    rng_seed: int | None = None,
) -> str:
    return generate_rna_sequence(
        motif=motif,
        num_candidates=num_candidates,
        min_total_length=min_total_length,
        max_total_length=max_total_length,
        rng_seed=rng_seed,
    )


@torch.inference_mode()
def fold_sequence_with_checkpoint(
    sequence: str,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> FoldResult:
    sequence = sequence.strip().upper().replace("T", "U")
    if not validate_rna_sequence(sequence):
        raise ValueError("sequence must contain only A, U, C, and G.")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    model_cfg = dict(checkpoint["config"]["model"])
    model_cfg.pop("type", None)
    model = RhoFoldModel(RhoFoldConfig(**model_cfg))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    input_ids = torch.tensor([encode_rna_sequence(sequence)], dtype=torch.long, device=device)
    padding_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    output = model(input_ids=input_ids, padding_mask=padding_mask, return_aux=True)
    return FoldResult(
        sequence=sequence,
        coords=output["coords"][0].detach().cpu(),
        plddt=output["plddt"][0].detach().cpu(),
    )


def save_fold_outputs(
    result: FoldResult,
    tensor_path: str | Path,
    pdb_path: str | Path | None = None,
) -> None:
    tensor_path = Path(tensor_path)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"sequence": result.sequence, "coords": result.coords, "plddt": result.plddt}, tensor_path)
    if pdb_path is not None:
        write_pdb(sequence=result.sequence, coords=result.coords, path=pdb_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 1D RNA from a motif and fold it with a local trained RhoFold checkpoint.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--motif", help="Fixed RNA motif used to generate a complete 1D scaffold sequence.")
    input_group.add_argument("--sequence", help="Complete RNA sequence to fold directly.")
    parser.add_argument("--checkpoint", required=True, help="Local checkpoint produced by train_3d.py.")
    parser.add_argument("--num-candidates", type=int, default=128)
    parser.add_argument("--min-total-length", type=int)
    parser.add_argument("--max-total-length", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", default="outputs/fold_3d.pt", help="Torch file containing sequence, coords, and pLDDT.")
    parser.add_argument("--output-pdb", help="Optional local PDB output path.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    sequence = args.sequence
    if sequence is None:
        sequence = generate_sequence_from_motif(
            motif=args.motif,
            num_candidates=args.num_candidates,
            min_total_length=args.min_total_length,
            max_total_length=args.max_total_length,
            rng_seed=args.seed,
        )
    result = fold_sequence_with_checkpoint(sequence, args.checkpoint, device=args.device)
    output_path = Path(args.output)
    save_fold_outputs(result, tensor_path=output_path, pdb_path=args.output_pdb)
    print(result.sequence)
    print(str(output_path))
    if args.output_pdb:
        print(str(Path(args.output_pdb)))


if __name__ == "__main__":
    main()
