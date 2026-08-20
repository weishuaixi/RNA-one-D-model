# RNA Scaffold Generator V2 Design

## 1. Goal

Replace the V1 scaffold generator with a clean V2 architecture focused on accurate motif-conditioned RNA scaffold generation. V2 deliberately breaks compatibility with V1 checkpoints and removes the independent total-length and motif-position heads.

V2 delivers five changes:

1. One canonical 1,815-sequence MMseqs2-clustered dataset contract.
2. Jointly valid left/right scaffold-length prediction.
3. Optional RNA-FM modes: none, frozen, and LoRA.
4. Confidence-based iterative remasking with diversity controls.
5. A reproducible benchmark covering meaningful baselines, ablations, structure metrics, and paired uncertainty estimates.

## 2. Non-goals

- Do not restore or train the removed in-house 3D model.
- Do not retain a V1 model implementation or silently migrate V1 checkpoints.
- Do not make RNAfold part of the differentiable training graph.
- Do not require RNA-FM or RNAfold for CPU unit tests.
- Do not claim biological function from sequence likelihood or RNAfold scores alone.

## 3. Versioning and compatibility

All new checkpoints contain `architecture_version: 2`. Checkpoint loading rejects missing or non-V2 architecture versions with an actionable error that says the model must be retrained.

The public CLI entry-point names remain stable:

- `train.py`
- `generate_scaffold.py`
- `validate_scaffolds.py`
- `benchmark_scaffolds.py`

Their V2 outputs and configuration schemas are intentionally incompatible with V1 where model semantics changed.

## 4. Canonical data contract

### 4.1 Files

The server training configuration uses:

- Sequence table: `data/processed/rna_merged_mmseqs80.csv`
- Cluster manifest: `outputs/mmseqs80_search/rna80_connected_components.tsv`

The expected release dataset contains 1,815 canonical, unique RNA sequences assigned to 1,185 MMseqs2 connected components. The known split contains 1,513 train, 133 validation, and 169 test sequences.

Paths remain configurable so the repository can run tests without committing the large server dataset. The default A800 server configuration points to the canonical paths under the repository instead of the older `train_sequences.csv`.

### 4.2 Audit and split rules

At setup, the data module validates:

- Every sequence contains only `A`, `U`, `C`, and `G` after `T` to `U` normalization.
- Sequence identifiers are unique.
- Normalized sequences are unique.
- Every eligible sequence has exactly one cluster assignment.
- A cluster appears in exactly one of train, validation, and test.
- The recorded split totals equal the eligible dataset total.

The split manifest records source SHA-256 hashes, clustering thresholds, random seed, record counts, cluster counts, and split assignments. Training stops on cluster leakage or incomplete assignments rather than falling back to a random row split.

Sequences longer than 512 nucleotides are retained in the source dataset and dynamically cropped during training. Model inputs and generation outputs never exceed 512 nucleotides.

## 5. V2 model

### 5.1 Backbone

The trainable scaffold generator remains a bidirectional Transformer encoder with learned RNA token embeddings, learned absolute positions, optional RNA-FM residue features, pre-norm encoder layers, and a four-base token head.

The A800 accuracy configuration remains:

- hidden size 768
- 12 attention heads
- 12 encoder layers
- feed-forward size 3,072
- dropout 0.1
- maximum length 512
- activation checkpointing enabled

### 5.2 Flank-length representation

V1's `length_head` and `position_head` are deleted. V2 predicts:

- `left_length_logits`: number of generated nucleotides before the motif
- `right_length_logits`: number of generated nucleotides after the motif

Each head has classes `0..max_length`. Selection is joint even though the marginal logits are produced separately. A pair `(left, right)` is legal only when:

```text
left >= min_flank_length
right >= min_flank_length
left + motif_length + right >= min_total_scaffold_length
left + motif_length + right <= requested_max_length
left + motif_length + right <= model.max_length
```

The pair score is the sum of left and right log probabilities. Training labels are derived exactly as:

```text
target_left_length = motif_start
target_right_length = target_length - motif_start - motif_length
```

Placement prediction receives a compact motif-only input so it cannot infer target geometry from padded canvas dimensions.

### 5.3 Loss

V2 optimizes:

```text
total_loss = base_loss
           + left_length_loss_weight * left_length_loss
           + right_length_loss_weight * right_length_loss
```

Base cross-entropy is evaluated only on prediction positions, never immutable motif positions. Label smoothing remains configurable. The default left and right weights are both 0.25.

## 6. RNA-FM integration

### 6.1 Modes

The pretrained configuration has three explicit modes:

- `none`: no RNA-FM encoder.
- `frozen`: load RNA-FM and freeze all its parameters.
- `lora`: freeze base RNA-FM parameters and train low-rank adapters.

The V1 `freeze` boolean is removed.

### 6.2 LoRA boundary

LoRA is injected into attention projection linear layers in the last four RNA-FM Transformer layers by default. Target discovery is validated against the loaded model; if no supported projections are found, construction fails with a diagnostic listing candidate module names.

Each adapter computes:

```text
y = frozen_linear(x) + scale * B(A(dropout(x)))
scale = alpha / rank
```

Defaults:

- rank 8
- alpha 16
- dropout 0.05
- last 4 layers

Only LoRA parameters are trainable inside RNA-FM. The project Transformer, RNA-FM-to-project projection, and output heads remain trainable. Metadata records RNA-FM mode, checkpoint hash when local, adapter targets, rank, alpha, dropout, and trainable parameter counts.

Tests use a small RNA-FM-shaped module and never download external weights.

## 7. V2 decoding

### 7.1 Canvas construction

The generator first runs the compact motif through the model and samples or selects one legal `(left, right)` pair. It constructs:

```text
[MASK] * left + motif + [MASK] * right
```

Motif tokens are restored after every iteration and can never be committed, remasked, or mutated.

### 7.2 Confidence remasking

Every iteration predicts all non-motif scaffold positions, including positions filled in a prior iteration. The schedule decreases the number of masked positions monotonically toward zero. After sampling bases, the lowest-confidence positions required by the schedule are changed back to `MASK`; high-confidence positions remain visible as context for the next pass.

The final pass resolves all scaffold positions. The reported normalized log probability is recalculated from final-token probabilities, not stale probabilities from the iteration in which a token was first committed.

### 7.3 Candidate diversity

Candidate generation enforces configurable controls:

- exact deduplication
- maximum pairwise identity or minimum normalized edit distance
- optional 5-mer similarity ceiling
- homopolymer-run ceiling or penalty
- soft GC target range
- diversified legal flank-pair sampling
- bounded generation attempts

The generator never duplicates an existing sequence to satisfy the requested count. If the diversity contract prevents the requested count, output reports requested, accepted, attempted, rejected, and rejection reasons.

## 8. Benchmark V2

### 8.1 Methods

The benchmark supports equal candidate budgets for:

- uniform nucleotide sampling
- first-order Markov
- second-order Markov
- Transformer without RNA-FM
- Transformer with frozen RNA-FM
- Transformer with RNA-FM LoRA (primary model)
- single-pass decoding ablation
- iterative-remasking decoding

Checkpoint-backed methods must load real V2 checkpoints. Missing checkpoints produce explicit unavailable results and cannot silently fall back to a heuristic.

### 8.2 Metrics

Per-candidate and aggregated metrics include:

- canonical validity
- exact motif preservation
- unique rate
- normalized edit diversity
- k-mer diversity and nearest-training k-mer similarity
- length and GC distributions
- maximum homopolymer run
- generation failures and rejection reasons
- elapsed time and peak CUDA memory when CUDA is used
- RNAfold status, MFE, paired fraction, and motif paired fraction when available

The report labels RNAfold-derived metrics as plausibility proxies, not proof of function.

### 8.3 Statistics and artifacts

Where methods share motif inputs, comparisons use motif-paired bootstrap confidence intervals. Each run writes atomically:

- `candidate_results.csv`
- `motif_summary.csv`
- `model_summary.csv`
- `paired_bootstrap.json`
- `run_manifest.json`
- `benchmark_report.md`

The run manifest records configuration, seeds, data hashes, checkpoint hashes, method status, software versions, and command arguments.

## 9. Configuration

The V2 A800 configuration uses the canonical dataset and cluster manifest, RNA-FM LoRA mode, BF16 mixed precision, gradient accumulation, activation checkpointing, early stopping, warmup-cosine scheduling, and best/last checkpoints.

Benchmark configuration declares each method independently and explicitly marks the RNA-FM LoRA iterative model as the primary model for tables and plots.

## 10. Error handling

V2 fails early with actionable messages for:

- V1 or unversioned checkpoints
- missing canonical data or cluster manifest
- cluster leakage or incomplete cluster assignment
- illegal flank targets
- unsupported RNA-FM module layout for LoRA
- requested motifs that cannot fit the configured constraints
- missing model checkpoints in formal benchmark runs

RNAfold remains an optional external validator and reports `ok`, `unavailable`, `timeout`, `failed`, or `parse_error` without crashing unrelated sequence-only evaluation.

## 11. Verification

Implementation follows test-driven development. Required coverage includes:

- flank-target derivation and all boundary constraints
- illegal joint pairs receiving zero selection probability
- motif-only placement input preventing geometry leakage
- LoRA injection, frozen base parameters, trainable adapter parameters, and metadata
- V2 checkpoint version rejection and acceptance
- monotonic remasking schedule and final resolution
- motif immutability through every decoding iteration
- diversity rejection accounting and no output duplication
- cluster-manifest completeness and cross-split leakage detection
- second-order Markov baseline behavior
- benchmark artifact schema and paired-bootstrap reproducibility
- CPU end-to-end smoke training, checkpoint loading, generation, and evaluation

The full existing test suite is migrated to V2 and must pass before completion. Formal RNA-FM and A800 training are performed on the server after the CPU test suite succeeds.
