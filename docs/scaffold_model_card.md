# RNA Scaffold Generator Model Card

## Model ownership

The project-owned model is a motif-conditioned bidirectional RNA scaffold generator. RNA-FM is an optional frozen external representation model. RNAfold and RhoFold+ are external validators and are not trained or redistributed by this project.

## Intended task

The input is a canonical A/U/C/G motif of at least four nucleotides. The model automatically selects a complete length and motif offset, preserves the motif exactly, and fills both flanks jointly. The model limit is 512 nucleotides.

## Architecture

The accuracy-oriented configuration uses a 12-layer Transformer with width 768, 12 attention heads, a 3072-dimensional feed-forward block, and optional projected 640-dimensional RNA-FM features. Separate heads predict scaffold bases, total length, and motif position. Generation uses confidence-ordered iterative masked denoising rather than independent one-shot base selection.

## Training data and split

The prepared server dataset contains 1,815 unique canonical RNA sequences merged from the Stanford/PDB-derived CSV and the validated all-atom cache. MMseqs2 search at 80% identity and 80% bidirectional coverage produced 1,185 connected-component families. The deterministic family-disjoint split contains 1,513 training, 133 validation, and 169 test records with zero family crossover. These counts describe the prepared server artifact and must be re-audited from its manifest before every full training run.

Long source RNAs are retained. Each epoch dynamically selects a window of at most 512 nucleotides and a new motif. Corruption mixes full scaffold masking, random masking, and contiguous span masking while motif tokens remain fixed.

## Optimization

Training uses AdamW, label-smoothed scaffold cross-entropy, length and position losses, gradient clipping, five-percent linear warmup, per-step cosine decay, early stopping, and best/last checkpoints. RNA-FM remains frozen in the primary configuration and must be compared against a no-RNA-FM checkpoint under the same budget.

## Evaluation

Required reports include canonical validity, exact motif preservation, uniqueness, diversity, nearest-training similarity, length and position errors, normalized likelihood, GC and complexity distributions, runtime, failures, and paired bootstrap intervals. RNAfold fields are reported only when the executable succeeds. MFE is never used as the sole ranking criterion.

## Limitations

- Sequence plausibility and RNAfold scores do not establish biological function.
- The training set is small for a 12-layer generator, so overfitting and memorization must be audited.
- RNA-FM benefit is an empirical ablation question, not an assumed contribution.
- Results above 512 nucleotides are unsupported.
- RhoFold+/experimental validation is required for strong tertiary-structure or functional claims.

## Reproducibility

Every generated row records seed, checkpoint SHA-256, sampling settings, motif coordinates, raw metrics, and status. Benchmark methods use identical motif, candidate, seed, and external-compute budgets. Missing predictions remain missing and are never replaced by outputs from another method.
