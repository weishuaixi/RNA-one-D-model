from __future__ import annotations

import argparse

from rna_scaffold.generate import (
    GenerationSettings,
    generate_candidates,
    write_candidates_fasta,
    write_candidates_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate RNA scaffolds from a trained checkpoint.")
    parser.add_argument("--motif", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fasta-output")
    parser.add_argument("--num-candidates", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--denoise-steps", type=int, default=12)
    parser.add_argument("--max-attempt-multiplier", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = GenerationSettings(
        num_candidates=args.num_candidates,
        max_length=args.max_length,
        seed=args.seed,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        denoise_steps=args.denoise_steps,
        max_attempt_multiplier=args.max_attempt_multiplier,
    )
    candidates = generate_candidates(args.checkpoint, args.motif, settings, args.device)
    write_candidates_jsonl(candidates, args.output)
    if args.fasta_output:
        write_candidates_fasta(candidates, args.fasta_output)


if __name__ == "__main__":
    main()
