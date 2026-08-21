# RNA Scaffold Generator V2 Model Card

## Model ownership and intended use

The project-owned model is a motif-conditioned bidirectional RNA scaffold
generator. Its input is a canonical A/U/C/G motif of at least four
nucleotides. It chooses legal left and right flank lengths, preserves the
motif exactly, and generates a complete sequence of at most 512 nucleotides.

RNA-FM is an external pretrained representation model. V2 supports no
RNA-FM, a frozen encoder, and LoRA adapters on a frozen encoder. RNAfold is
an optional external validator, not a project model or a differentiable
training component.

## Architecture

The accuracy configuration uses a 12-layer project Transformer with width
768, 12 attention heads, a 3,072-dimensional feed-forward block, and
optional projected 640-dimensional RNA-FM residue features. Separate heads
predict left and right flank lengths. Their scores are selected jointly
under the requested maximum length, model maximum length, minimum total
scaffold length, and per-flank minimum.

The primary RNA-FM configuration applies rank-8 LoRA with alpha 16 and input
dropout 0.05 to q/k/v/out attention projections in the final four RNA-FM
layers. Base RNA-FM parameters remain frozen. Project-model parameters,
RNA-FM-to-project projection, output heads, and LoRA parameters remain
trainable.

Generation starts from
`[MASK] * left + motif + [MASK] * right`. Every iteration re-predicts all
scaffold positions, then re-masks the lowest-confidence positions required
by a monotonic schedule. Motif tokens are restored after every iteration
and never sampled, committed, or re-masked. Final normalized likelihood is
calculated from a fresh forward pass over the fully resolved sequence.

## Training data and split

The canonical release contract uses:

- `data/processed/rna_merged_mmseqs80.csv`
- `outputs/mmseqs80_search/rna80_connected_components.tsv`

The prepared artifact contains 1,815 canonical unique sequences assigned to
1,185 MMseqs2 connected components at 80% identity and 80% bidirectional
coverage. The deterministic cluster-disjoint split contains 1,513 training,
133 validation, and 169 test records.

Setup rejects invalid bases, duplicate IDs, duplicate normalized sequences,
missing or extra cluster assignments, declared cluster-size mismatches, and
cross-partition cluster leakage. It records source SHA-256 values, split
counts, cluster counts, thresholds, seed, and assignments. Long source RNAs
are retained and dynamically cropped to the 512-nucleotide model window.

## Optimization and checkpoint compatibility

Training uses AdamW, label-smoothed base cross-entropy on prediction
positions only, left- and right-length cross-entropy weights of 0.25,
gradient clipping, 5% linear warmup, per-step cosine decay, early stopping,
activation checkpointing, BF16 mixed precision, and best/last checkpoints.

Every accepted checkpoint has integer `architecture_version: 2`.
Unversioned checkpoints, non-integer versions, and versions other than 2
are rejected with an instruction to retrain. V2 does not silently convert
or heuristically replace an incompatible checkpoint.

Checkpoint metadata records RNA-FM mode, local RNA-FM checkpoint SHA-256,
adapter targets and hyperparameters, and total/trainable parameter counts.

## Generation controls

Candidate generation supports exact deduplication, minimum normalized edit
distance, optional k-mer similarity ceiling, homopolymer ceiling, soft or
enforced GC bounds, diversified legal flank sampling, and a bounded attempt
budget. A shortfall reports requested, attempted, accepted, rejected, and
per-reason rejection counts; candidates are never duplicated to fill a
quota.

## Evaluation

Formal comparisons use equal candidate budgets for uniform, first-order
Markov, second-order Markov, no-RNA-FM Transformer, frozen-RNA-FM
Transformer, RNA-FM-LoRA Transformer, single-pass decoding, and iterative
remasking. Checkpoint-backed methods must load real V2 checkpoints. Missing
checkpoints and inference errors produce explicit terminal method states
and zero fabricated predictions.

Required sequence metrics include canonical validity, exact motif
preservation, unique rate, normalized edit and k-mer diversity,
nearest-training k-mer similarity, length and GC distributions, maximum
homopolymer run, rejection/failure accounting, elapsed time, and peak CUDA
memory where applicable. Motif-paired bootstrap intervals quantify method
differences.

RNAfold statuses are `ok`, `unavailable`, `timeout`, `failed`, or
`parse_error`; `not_run` means no structure attempt was scheduled.
MFE, paired fraction, and motif paired fraction are plausibility proxies,
not evidence of function.

Each successful benchmark run starts from a new or empty directory and
writes exactly:

- `candidate_results.csv`
- `motif_summary.csv`
- `model_summary.csv`
- `paired_bootstrap.json`
- `run_manifest.json`
- `benchmark_report.md`

## Limitations

- Sequence likelihood and RNAfold outputs do not establish biological
  function.
- The canonical training corpus is small for the primary model size, so
  memorization and generalization require explicit audit.
- RNA-FM benefit must be shown by matched-budget ablation.
- Outputs longer than 512 nucleotides are unsupported.
- Experimental validation is required before biological claims.

## Release status

The CPU release gate proves training-path execution, V2 checkpoint
round-trip, generation of at least two unique motif-preserved candidates,
and sequence-only evaluation. It is runtime evidence, not model-quality
evidence. Final quality claims require full A800 training and the formal
paired benchmark with all declared V2 checkpoints.
