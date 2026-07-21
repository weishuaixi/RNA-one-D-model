from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from rna_scaffold.generate import build_auto_masked_scaffold_prompts, generate_rna_sequence
from rna_scaffold.utils import validate_rna_sequence
from rna_scaffold_3d.rhofold import RhoFoldConfig, RhoFoldModel
from rna_scaffold_3d.pdb_writer import write_pdb
from rna_scaffold_3d.sequence import RNA3D_MASK_ID, RNA_ID_TO_BASE, encode_rna_sequence


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

    model, _ = _load_model(checkpoint_path, device)
    return _fold_with_model(sequence, model, device)


@torch.inference_mode()
def fold_motif_with_checkpoint(
    motif: str,
    checkpoint_path: str | Path,
    num_candidates: int = 8,
    min_total_length: int | None = None,
    max_total_length: int | None = None,
    rng_seed: int | None = None,
    denoise_steps: int = 6,
    device: str | torch.device = "cpu",
) -> FoldResult:
    motif = motif.strip().upper().replace("T", "U")
    if not validate_rna_sequence(motif):
        raise ValueError("motif must contain only A, U, C, and G.")
    if denoise_steps <= 0:
        raise ValueError("denoise_steps must be positive.")
    model, has_joint_sequence_head = _load_model(checkpoint_path, device)
    if not has_joint_sequence_head or model.config.vocab_size <= RNA3D_MASK_ID:
        raise ValueError("Motif generation requires a checkpoint trained with the joint 1D/3D objective.")

    prompts = build_auto_masked_scaffold_prompts(
        motif=motif,
        num_candidates=num_candidates,
        min_total_length=min_total_length,
        max_total_length=max_total_length,
        rng_seed=rng_seed,
    )
    best: FoldResult | None = None
    best_confidence = float("-inf")
    motif_ids = torch.tensor(encode_rna_sequence(motif), dtype=torch.long, device=device)
    for prompt in prompts:
        input_ids = torch.full(
            (1, prompt.total_length),
            RNA3D_MASK_ID,
            dtype=torch.long,
            device=device,
        )
        start = prompt.motif_start
        input_ids[0, start : start + len(motif)] = motif_ids
        padding_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        predicted_ids = _iterative_decode_sequence(
            model=model,
            input_ids=input_ids,
            padding_mask=padding_mask,
            denoise_steps=denoise_steps,
        )[0]
        predicted_ids[start : start + len(motif)] = motif_ids
        sequence = "".join(RNA_ID_TO_BASE[int(index)] for index in predicted_ids.tolist())
        result = _fold_with_model(sequence, model, device)
        confidence = float(result.plddt.mean().item())
        if confidence > best_confidence:
            best = result
            best_confidence = confidence
    if best is None:  # pragma: no cover - prompt validation guarantees candidates
        raise RuntimeError("No motif scaffold candidates were generated.")
    return best


def _iterative_decode_sequence(
    model: RhoFoldModel,
    input_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    denoise_steps: int,
) -> torch.Tensor:
    tokens = input_ids.clone()
    original_scaffold = tokens.eq(RNA3D_MASK_ID)
    total_scaffold = original_scaffold.sum(dim=1)
    for step in range(denoise_steps):
        remaining = tokens.eq(RNA3D_MASK_ID)
        if not remaining.any():
            break
        output = model(input_ids=tokens, padding_mask=padding_mask, return_aux=True)
        probabilities = torch.softmax(output["sequence_logits"][..., 1:5], dim=-1)
        confidence, predicted = probabilities.max(dim=-1)
        predicted = predicted + 1
        for row in range(tokens.size(0)):
            remaining_positions = torch.nonzero(remaining[row], as_tuple=False).flatten()
            if remaining_positions.numel() == 0:
                continue
            target_filled = math.ceil(int(total_scaffold[row].item()) * (step + 1) / denoise_steps)
            already_filled = int(total_scaffold[row].item()) - int(remaining_positions.numel())
            fill_count = min(
                int(remaining_positions.numel()),
                max(1, target_filled - already_filled),
            )
            ranked = torch.topk(confidence[row, remaining_positions], k=fill_count).indices
            chosen = remaining_positions[ranked]
            tokens[row, chosen] = predicted[row, chosen]
    if tokens.eq(RNA3D_MASK_ID).any():
        output = model(input_ids=tokens, padding_mask=padding_mask, return_aux=True)
        predicted = output["sequence_logits"][..., 1:5].argmax(dim=-1) + 1
        tokens = torch.where(tokens.eq(RNA3D_MASK_ID), predicted, tokens)
    return tokens


def _load_model(
    checkpoint_path: str | Path,
    device: str | torch.device,
) -> tuple[RhoFoldModel, bool]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = checkpoint["model_state_dict"]
    model_cfg = dict(checkpoint["config"]["model"])
    model_cfg.pop("type", None)
    model_cfg.setdefault("vocab_size", int(state["seq_embedder.embedding.weight"].shape[0]))
    model = RhoFoldModel(RhoFoldConfig(**model_cfg))
    has_joint_sequence_head = any(key.startswith("sequence_head.") for key in state)
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(device)
    return model, has_joint_sequence_head


def _fold_with_model(
    sequence: str,
    model: RhoFoldModel,
    device: str | torch.device,
) -> FoldResult:
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
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--min-total-length", type=int)
    parser.add_argument("--max-total-length", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--denoise-steps", type=int, default=6)
    parser.add_argument("--output", default="outputs/fold_3d.pt", help="Torch file containing sequence, coords, and pLDDT.")
    parser.add_argument("--output-pdb", help="Optional local PDB output path.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.sequence is None:
        result = fold_motif_with_checkpoint(
            motif=args.motif,
            checkpoint_path=args.checkpoint,
            num_candidates=args.num_candidates,
            min_total_length=args.min_total_length,
            max_total_length=args.max_total_length,
            rng_seed=args.seed,
            denoise_steps=args.denoise_steps,
            device=args.device,
        )
    else:
        result = fold_sequence_with_checkpoint(args.sequence, args.checkpoint, device=args.device)
    output_path = Path(args.output)
    save_fold_outputs(result, tensor_path=output_path, pdb_path=args.output_pdb)
    print(result.sequence)
    print(str(output_path))
    if args.output_pdb:
        print(str(Path(args.output_pdb)))


if __name__ == "__main__":
    main()
