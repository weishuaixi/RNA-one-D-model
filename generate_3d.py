from __future__ import annotations

import argparse
import json

from rna_scaffold.structure import generate_rna_and_prepare_3d


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an RNA sequence from a motif and prepare optional 3D prediction.")
    parser.add_argument("--motif", required=True, help="Fixed RNA motif, e.g. GCGG.")
    parser.add_argument("--output-dir", default="outputs/rna_3d")
    parser.add_argument("--name", default="rna_candidate")
    parser.add_argument("--num-candidates", type=int, default=128)
    parser.add_argument("--min-total-length", type=int)
    parser.add_argument("--max-total-length", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--predictor-command",
        nargs="+",
        help=(
            "Optional RhoFold/RhoFold+ command template. Use {fasta}, "
            "{output_pdb}, and {output_dir} placeholders."
        ),
    )
    args = parser.parse_args()

    result = generate_rna_and_prepare_3d(
        motif=args.motif,
        output_dir=args.output_dir,
        name=args.name,
        predictor_command=args.predictor_command,
        num_candidates=args.num_candidates,
        min_total_length=args.min_total_length,
        max_total_length=args.max_total_length,
        rng_seed=args.seed,
    )
    print(result.sequence)
    print(
        json.dumps(
            {
                "status": result.status.value,
                "fasta_path": str(result.fasta_path),
                "structure_path": str(result.structure_path) if result.structure_path else None,
                "message": result.message,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
