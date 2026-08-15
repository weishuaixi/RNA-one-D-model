# RNA motif-conditioned scaffold generator redesign

Date: 2026-08-15

## 1. Objective

The project will focus on one core contribution: generating complete RNA scaffold sequences around a fixed functional motif. The user provides a contiguous RNA motif of 4–64 nucleotides. The system automatically chooses the complete sequence length and motif position, generates both flanks, and returns ranked scaffold candidates no longer than 512 nucleotides.

The locally trained 3D predictor is removed from the product and evaluation path. ViennaRNA RNAfold and the official RhoFold+ implementation are external validators, not trainable components owned by this project.

## 2. Success criteria

The generator must:

- preserve every motif nucleotide exactly in every valid candidate;
- support variable motif lengths from 4 through 64 nucleotides;
- learn a length distribution and stop automatically, with a hard maximum of 512 nucleotides;
- generate left and right flanks jointly rather than sampling them independently;
- produce multiple reproducible candidates under a fixed random seed;
- expose generation likelihood, validity, novelty, diversity, and structural-validation fields;
- use family-disjoint train, validation, and test partitions;
- outperform random, first-order Markov, and non-pretrained Transformer baselines on a preregistered evaluation suite;
- report failures and uncertainty rather than selecting only successful examples.

## 3. System boundary

### Owned by this project

- data ingestion, normalization, deduplication, and split manifests;
- motif-conditioned length and position prediction;
- motif-protected bidirectional scaffold generation;
- candidate sampling and deterministic replay;
- novelty, diversity, and sequence-quality evaluation;
- candidate ranking and structured result files;
- reproducible benchmark and ablation workflows.

### External dependencies

- RNA-FM supplies optional pretrained RNA representations;
- ViennaRNA RNAfold supplies secondary-structure predictions and thermodynamic features;
- official RhoFold+ supplies tertiary-structure predictions for a small final candidate set.

External tools must be invoked through adapters. The generator remains usable when either structural validator is unavailable; missing validation is recorded explicitly and never silently replaced with a fabricated score.

## 4. Data design

### Sources

- the existing Stanford/PDB-derived RNA sequences;
- public Rfam families;
- additional public PDB RNA chains when their provenance and release date are available.

Every record stores source, accession, family, release metadata, normalized sequence, and an immutable content hash.

### Normalization

- convert T to U;
- accept A/U/C/G for the initial generator;
- reject or separately record ambiguous bases;
- remove exact duplicate sequences;
- retain provenance for all merged duplicate records;
- enforce a training length range compatible with the 512-nucleotide inference limit.

### Leakage prevention

Families are the primary split unit. All members of an Rfam family remain in one partition. Records without family labels are clustered by sequence similarity before assignment. Exact sequence overlap across partitions is forbidden. Near-duplicate audits and split manifests are saved and validated before training and evaluation.

### Self-supervised scaffold examples

For every training sequence, each epoch samples a contiguous motif with length 4–64 at a non-fixed position. At least one nucleotide must remain outside the motif. The target is the original full sequence. Sampling covers terminal, central, short-flank, long-flank, and asymmetric placements rather than always selecting the center.

## 5. Model architecture

### Pretrained motif/context representation

RNA-FM is an optional pretrained initializer. It is first frozen and compared with a randomly initialized encoder under an identical generator and training budget. LoRA or partial unfreezing is permitted only as a separate ablation. RNA-FM is retained in the final model only if it improves held-out-family results with confidence intervals that exclude a negligible effect.

### Length and motif-position head

The encoder representation predicts a joint categorical distribution over valid `(total_length, motif_start)` pairs. Invalid combinations are masked. Training uses the known full length and sampled motif position. Inference samples or ranks valid pairs, and never exceeds 512 nucleotides.

### Motif-protected denoising generator

For a selected length and position, the input canvas contains mask tokens on both flanks and immutable motif tokens in the selected interval. A bidirectional Transformer predicts masked bases using both flanks and motif context. Iterative denoising commits high-confidence bases first and revisits uncertain positions. Motif positions are overwritten with the original motif after every iteration and verified before returning a candidate.

The generator predicts A/U/C/G only at scaffold positions. It records token probabilities and normalized log-likelihood. Sampling supports temperature, top-k, and top-p controls.

### Training objectives

The initial supervised objective contains:

- masked nucleotide cross-entropy;
- joint length/position cross-entropy;
- optional confidence calibration loss;
- balanced sampling across sequence length and motif placement.

A second structure-preference stage compares candidates from the same motif using RNAfold-derived preferences. Preference optimization must not reward MFE alone, because that encourages GC-rich or repetitive artifacts. The reward combines normalized ensemble free energy, base-pairing statistics, ensemble diversity, sequence complexity, motif constraints when provided, novelty, and validity. Reward components and weights are logged and ablated.

## 6. Generation and ranking

The public generation API accepts motif, candidate count, seed, and sampling controls. Total length remains automatic. A default run produces 256 candidates; benchmark budgets are identical for all methods.

Candidate processing occurs in this order:

1. hard validity and motif-preservation checks;
2. exact duplicate collapse;
3. model likelihood and confidence calculation;
4. training-set nearest-neighbour and intra-batch diversity calculation;
5. RNAfold validation when installed;
6. composite ranking with all component scores retained;
7. optional RhoFold+ validation for the top configurable subset.

The result format contains the full sequence, flanks, motif coordinates, length, seed, generator checkpoint hash, raw metrics, ranking components, external-tool versions, and failure fields.

## 7. Evaluation

### Baselines

- uniform random nucleotide sampling;
- first-order Markov sampling learned from training data;
- the existing randomly initialized Encoder–Decoder Transformer;
- the new generator without RNA-FM;
- the new generator without structure-preference training;
- the complete model.

### Primary metrics

- valid-sequence rate;
- exact motif-preservation rate;
- valid-scaffold success rate after all preregistered filters;
- RNAfold pass rate and component distributions;
- RhoFold+ confidence and geometry metrics on the same selected budget;
- nearest-training-sequence similarity;
- unique-candidate rate and pairwise diversity;
- length-distribution calibration;
- inference time and peak memory.

### Statistical reporting

All methods use the same held-out motifs, candidate count, random seeds, and structural-compute budget. Reports include per-target rows, mean, median, bootstrap confidence intervals, paired differences, failure counts, and full configuration hashes. The benchmark never substitutes missing predictions with successful examples from another method.

## 8. Interfaces

The primary command will generate a machine-readable candidate file and optional FASTA:

```text
python generate_scaffold.py \
  --motif GCGG \
  --checkpoint checkpoints/scaffold_best.ckpt \
  --num-candidates 256 \
  --max-length 512 \
  --seed 42 \
  --output outputs/GCGG_candidates.jsonl
```

Structural validation is a separate command so training and sequence generation do not depend on local RNAfold or RhoFold+ installation:

```text
python validate_scaffolds.py \
  --input outputs/GCGG_candidates.jsonl \
  --rnafold \
  --rhofold-top-k 10
```

## 9. Error handling

- invalid motifs fail before model loading;
- impossible length/position distributions fail with diagnostic probabilities;
- candidates that reach the 512-nucleotide boundary without a valid termination state are marked truncated;
- external-tool timeouts and non-zero exits are recorded per candidate;
- checkpoint, tokenizer, dataset, and split-manifest versions are checked before use;
- partial result files are written atomically and can be resumed.

## 10. Removal of the local 3D subsystem

The following project-owned components are deleted:

- `rna_scaffold_3d/`;
- `train_3d.py`, `fold_3d.py`, and `evaluate_3d.py`;
- `configs/train_3d_*.yaml`;
- local-3D training and evaluation scripts;
- local-3D tests;
- `checkpoints_3d/` and all nested local-3D weights;
- temporary predictions and reports whose subject is the local 3D model;
- packaging entry points and documentation for local 3D training or inference.

External RhoFold+ adapters and future external-structure benchmark results are not part of the deleted subsystem.

## 11. Verification strategy

Implementation follows tests first. Unit tests cover motif immutability, length/position masking, variable motifs, deterministic sampling, termination, deduplication, and adapter failures. Integration tests train a tiny model on synthetic RNA, generate candidates, run a mocked RNAfold adapter, and verify the result schema. Dataset tests intentionally inject family, exact-sequence, and near-duplicate leakage and require fail-closed behavior.

Before claiming completion, the project must pass the one-dimensional test suite, command-line smoke tests, a tiny overfit test, and a reproducibility run. External validators receive separate installation and smoke-test reports.

## 12. Delivery sequence

1. remove the local 3D subsystem and clean package metadata;
2. establish leakage-safe sequence datasets and manifests;
3. implement the length/position head and motif-protected denoising generator;
4. implement checkpoint-based generation and structured candidate output;
5. add RNAfold and RhoFold+ adapters;
6. implement benchmarks, ablations, and statistical reports;
7. train small and full configurations, then publish verified results.
