# RNA Scaffold Generation Completion Design

Date: 2026-08-19

## 1. Goal and scope

Complete the existing motif-conditioned one-dimensional RNA scaffold system so that training, checkpoint inference, candidate generation, external structural validation, benchmarking, and ablation form one reproducible path. The owned scientific contribution remains the trainable scaffold generator. RNA-FM is a frozen external feature provider, while RNAfold and RhoFold+ are optional external validators.

This work does not train or restore a project-owned 3D model. It also does not delete or modify unrelated legacy 3D files, checkpoints, server archives, or the user's current uncommitted files.

## 2. Required behavior

- The formal generation API requires a compatible trained checkpoint. Missing or incompatible checkpoints fail with an actionable error.
- The historical first-order sequence prior remains available only through an explicitly named Markov baseline API and command.
- An input motif contains only A/U/C/G, has at least four nucleotides, and must leave room for the configured minimum scaffold context within the 512-nucleotide model limit.
- Total length and motif position are automatic by default. Invalid `(length, position)` combinations are masked before selection.
- Every generated candidate preserves the motif exactly and contains only A/U/C/G.
- Equal seeds, model files, settings, software, and hardware produce reproducible results.
- External validators are optional. Their absence, timeout, parse failure, and non-zero exit are recorded rather than replaced with invented values.

## 3. Training path

Each epoch samples a fresh motif and window from every eligible RNA record. Motif length is dynamically sampled from four nucleotides through the largest value feasible for that sequence and the 512-nucleotide training window. Sampling retains short motifs while also covering longer functional contexts. Long source RNAs are retained and dynamically cropped.

Training corruptions mix full-flank masking, random token masking, and contiguous span masking. The corruption distribution includes easy partial reconstruction and the fully masked inference condition. Motif tokens remain immutable and are excluded from scaffold-token loss.

The trainable generator uses token, absolute-position, and frozen RNA-FM features followed by the existing 12-layer bidirectional Transformer. It predicts scaffold bases, total length, and motif position. Candidate confidence is derived from calibrated token probabilities; the current unsupervised confidence head is removed rather than exposed as a meaningful score.

Optimization uses AdamW, gradient clipping, linear warmup over five percent of optimizer steps, then cosine decay to two percent of the initial learning rate. Scheduler state is checkpointed and updated per optimizer step. Validation monitors total loss and component metrics. Training saves best and last checkpoints and supports early stopping and explicit resume.

## 4. Checkpoint-backed generation

Generation loads the Lightning checkpoint, validates tokenizer/model metadata, records the checkpoint SHA-256, and runs the learned model. It predicts length and position distributions and ranks or samples only valid pairs.

For each selected canvas, iterative denoising repeatedly predicts unresolved scaffold positions. Temperature plus optional top-k/top-p filtering supplies diversity. At every iteration, the most confident unresolved fraction is committed, motif tokens are restored, and invalid token classes are excluded. Candidate log probability is the normalized sum of committed token log probabilities. Generation has bounded attempts, reports unique-candidate shortfall explicitly, and never silently switches to the Markov baseline.

The public result contains the complete sequence, both flanks, motif coordinates, total length, normalized model log probability, generation controls, seed, checkpoint hash, validity fields, ranking components, and failure status. JSONL and optional FASTA writes are atomic.

## 5. Candidate filtering and external validation

Hard checks run before ranking: canonical alphabet, exact motif preservation, valid coordinates, length limit, and duplicate collapse. Sequence-only metrics include GC fraction, homopolymer burden, low-complexity penalty, normalized likelihood, training-set nearest-neighbour similarity, and intra-batch diversity.

RNAfold is invoked through a subprocess adapter without `shell=True`. Its raw dot-bracket structure, MFE, paired fraction, motif paired fraction, runtime, version, and status are retained. Composite ranking uses normalized components and never treats MFE alone as quality. RhoFold+ remains a separate optional top-k validator with the same explicit status semantics.

## 6. Evaluation and ablation

All methods receive identical held-out motifs, candidate budgets, seeds, and external-compute budgets. Required methods are uniform random, first-order Markov, learned model without RNA-FM, learned model with RNA-FM, single-pass decoding, iterative decoding, and iterative decoding with RNAfold reranking.

Reports include per-candidate CSV/JSONL, per-motif summaries, aggregate JSON, configuration and checkpoint hashes, failure counts, and paired bootstrap 95% confidence intervals. Metrics cover motif preservation, canonical validity, unique rate, length and position error, token loss/accuracy, normalized likelihood, novelty, diversity, GC/complexity distributions, RNAfold availability and structural components, runtime, and peak memory.

## 7. Interfaces

The primary command is checkpoint-only:

```text
python generate_scaffold.py \
  --motif GCGG \
  --checkpoint checkpoints_scaffold_a800_mmseqs80/best.ckpt \
  --num-candidates 256 \
  --max-length 512 \
  --seed 42 \
  --output outputs/GCGG_candidates.jsonl
```

The Markov implementation is invoked explicitly as a baseline. It is never a fallback for the primary command.

Structural validation remains separate:

```text
python validate_scaffolds.py \
  --input outputs/GCGG_candidates.jsonl \
  --rnafold \
  --output outputs/GCGG_validated.jsonl
```

Benchmark and ablation execution use one configuration file and produce a run manifest containing exact method budgets and artifact hashes.

## 8. Compatibility and migration

The old convenience call that generated without a checkpoint is intentionally broken for scientific correctness. Users receive an error directing them either to the checkpoint-backed API or the explicitly named Markov baseline. Existing training checkpoints remain loadable when their stored hyperparameters are sufficient; incompatibility is reported before generation. No user-owned checkpoint or legacy file is deleted.

## 9. Verification

Every production behavior is introduced test-first. Unit tests cover checkpoint requirements, metadata validation, valid length-position selection, motif immutability, sampling filters, iterative commitment, deterministic seeds, candidate shortfall, atomic output, scheduler steps, corruption modes, RNAfold parsing/failures, metrics, and paired bootstrap reproducibility.

Integration tests train or construct a tiny CPU checkpoint, generate candidates twice, validate through a fake RNAfold executable, and run a smoke benchmark. Completion requires the entire local test suite, CLI help checks, deterministic replay, a tiny overfit test, and a release audit that separates locally verified facts from server-only training results.

## 10. Delivery order

1. checkpoint loading and strict public API;
2. iterative denoising and structured candidate output;
3. corruption curriculum, calibrated likelihood, scheduler, callbacks, and resume;
4. RNAfold validation and candidate ranking;
5. baselines, ablations, statistical reports, and smoke benchmark;
6. documentation, server configurations, and final verification audit.
