# RNA Scaffold Generation Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the existing RNA scaffold project with strict checkpoint-backed iterative generation, improved denoising training, optional RNAfold ranking, and reproducible baselines/ablations.

**Architecture:** Keep the existing bidirectional `MotifDenoisingTransformer` and leakage-safe datasets. Add focused modules for checkpoint loading, iterative decoding, sequence scoring, external validation, and evaluation; the formal API never falls back to the retained Markov baseline.

**Tech Stack:** Python 3.10+, PyTorch 2.1+, Lightning 2.3+, pytest 8+, PyYAML, optional RNA-FM and ViennaRNA CLI.

**Spec:** `docs/superpowers/specs/2026-08-19-scaffold-generation-completion-design.md`

## Global Constraints

- Formal generation requires a compatible Lightning checkpoint.
- Markov generation is exposed only as an explicitly named baseline.
- Motifs contain only A/U/C/G, are at least 4 nt, and must fit the 512 nt canvas with configured scaffold context.
- Motif tokens are immutable during every decoding iteration.
- Generated scaffold positions contain only A/U/C/G.
- External validators are optional subprocess adapters and never fabricate missing scores.
- User-owned legacy 3D files, checkpoints, archives, and unrelated dirty worktree files are untouched.
- Every new behavior is developed red-green-refactor with deterministic CPU tests.

---

### Task 1: Add checkpoint loading and iterative decoding primitives

**Files:**
- Create: `rna_scaffold/checkpoints.py`
- Create: `rna_scaffold/decoding.py`
- Modify: `rna_scaffold/model.py`
- Test: `tests/test_checkpoint_loading.py`
- Test: `tests/test_iterative_decoding.py`

**Interfaces:**
- Produces: `LoadedScaffoldModel(model, tokenizer, checkpoint_sha256, max_length)`.
- Produces: `load_scaffold_checkpoint(path, device="cpu") -> LoadedScaffoldModel`.
- Produces: `select_length_position(output, motif_length, max_length, generator, sample) -> tuple[int, int]`.
- Produces: `iterative_denoise(model, tokenizer, motif, total_length, motif_start, settings, generator, device) -> DecodedScaffold`.

- [ ] **Step 1: Write checkpoint contract tests**

Tests construct a tiny `RnaScaffoldLitModule`, save a Lightning-shaped dictionary containing `state_dict` and `hyper_parameters`, and assert that loading reconstructs the model, tokenizer, maximum length, and SHA-256. A missing path must raise `FileNotFoundError`; missing model hyperparameters must raise `CheckpointCompatibilityError`.

- [ ] **Step 2: Run the checkpoint tests and verify RED**

Run: `pytest tests/test_checkpoint_loading.py -v`

Expected: collection failure because `rna_scaffold.checkpoints` does not exist.

- [ ] **Step 3: Implement strict checkpoint reconstruction**

Use `torch.load(path, map_location=device, weights_only=False)`, accept Lightning `state_dict` keys, reconstruct `RnaScaffoldLitModule` only from saved hyperparameters, call `load_state_dict(..., strict=True)`, set eval mode, and hash the exact checkpoint bytes. Do not download RNA-FM when the checkpoint metadata says `kind: none`; preserve the configured RNA-FM path behavior otherwise.

- [ ] **Step 4: Write iterative decoding tests**

Use a deterministic tiny model whose logits prefer known bases. Assert valid length/position masking, exact motif restoration after each step, monotonically decreasing unresolved positions, canonical output, normalized log probability, top-k/top-p filtering, and identical results for equal seeds.

- [ ] **Step 5: Run decoding tests and verify RED**

Run: `pytest tests/test_iterative_decoding.py -v`

Expected: collection failure because decoding interfaces do not exist.

- [ ] **Step 6: Implement valid pair selection and MaskGIT-style decoding**

Combine `log_softmax(length_logits)` and `log_softmax(position_logits)` over pairs satisfying `motif_length < total_length <= max_length` and `0 <= motif_start <= total_length - motif_length`. Iterative decoding predicts unresolved positions, filters four-base logits by temperature/top-k/top-p, samples bases, commits `ceil(unresolved / remaining_steps)` highest-confidence positions, restores motif IDs, and returns per-position committed log probabilities.

- [ ] **Step 7: Remove the unsupervised confidence head**

Remove `confidence` from `ScaffoldModelOutput` and `confidence_head` from the model. Compatibility errors must explain that old checkpoints containing these extra keys require the migration loader to discard only `model.confidence_head.weight` and `model.confidence_head.bias`; all other key mismatches remain fatal.

- [ ] **Step 8: Run focused and regression tests**

Run: `pytest tests/test_checkpoint_loading.py tests/test_iterative_decoding.py tests/test_denoising_model.py tests/test_model.py -v`

Expected: PASS.

### Task 2: Replace the public generator with strict learned-model generation

**Files:**
- Rewrite: `rna_scaffold/generate.py`
- Modify: `rna_scaffold/__init__.py`
- Create: `generate_scaffold.py`
- Test: `tests/test_generate.py`
- Create: `tests/test_generate_cli.py`

**Interfaces:**
- Produces: `GenerationSettings(num_candidates=256, max_length=512, seed=42, temperature=1.0, top_k=None, top_p=0.95, denoise_steps=12, max_attempt_multiplier=8)`.
- Produces: `ScaffoldCandidate` with sequence, flanks, motif coordinates, likelihood, checkpoint hash, settings, validity, and status.
- Produces: `generate_candidates(checkpoint, motif, settings, device="cpu") -> list[ScaffoldCandidate]`.
- Produces: `generate_rna_sequence(motif, checkpoint, **settings) -> str`.
- Retains: `generate_markov_baseline(motif, train_data, seed, min_total_length=None, max_total_length=None) -> ScaffoldResult`.

- [ ] **Step 1: Replace old API tests with strict behavior tests**

Assert that `generate_rna_sequence("GCGG")` raises `TypeError` or a checkpoint-required error, a tiny checkpoint generates canonical sequences with exact motifs, unique candidates are deterministic, and Markov output is available only through `generate_markov_baseline`.

- [ ] **Step 2: Run generation tests and verify RED**

Run: `pytest tests/test_generate.py -v`

Expected: failures because the current API silently uses `RnaTrainingPrior`.

- [ ] **Step 3: Implement candidate orchestration and sequence metrics**

Validate motif before loading the checkpoint. Repeatedly select valid canvases and call iterative decoding until the unique target is met or the bounded attempt budget expires. Compute GC fraction, maximum homopolymer length, Shannon base entropy, normalized log probability, and exact motif preservation. Return fewer candidates with `status="shortfall"` rather than duplicating sequences.

- [ ] **Step 4: Implement atomic JSONL/FASTA writers and CLI**

`generate_scaffold.py` requires `--motif`, `--checkpoint`, and `--output`; exposes sampling controls and `--device`; writes a sibling temporary file followed by `Path.replace`. Optional `--fasta-output` uses stable candidate IDs and the same atomic rule.

- [ ] **Step 5: Write and run CLI tests**

Run: `pytest tests/test_generate.py tests/test_generate_cli.py -v`

Expected: PASS for help, invalid motif, missing checkpoint, deterministic tiny-checkpoint generation, JSONL schema, and atomic replacement.

### Task 3: Improve corruption training and optimizer scheduling

**Files:**
- Modify: `rna_scaffold/data.py`
- Modify: `rna_scaffold/datamodule.py`
- Modify: `rna_scaffold/lightning_module.py`
- Modify: `train.py`
- Modify: `configs/train_scaffold_a800.yaml`
- Test: `tests/test_dataset.py`
- Test: `tests/test_training_schedule.py`
- Test: `tests/test_training_config.py`

**Interfaces:**
- Adds data options: `full_mask_probability`, `min_random_mask_fraction`, `max_random_mask_fraction`, `span_mask_probability`, `mean_span_length`.
- Adds model options: `warmup_fraction=0.05`, `min_lr_fraction=0.02`, `label_smoothing=0.05`.
- Adds trainer options outside Lightning args: `early_stopping_patience`, `resume_from_checkpoint`.

- [ ] **Step 1: Write corruption curriculum tests**

For fixed seeds, assert motif tokens are always fixed, supervised scaffold positions are masked, partial corruption leaves some non-motif context visible, full-mask mode reproduces inference, span mode produces at least one adjacent masked pair, and `set_epoch` changes corruption deterministically.

- [ ] **Step 2: Run dataset tests and verify RED**

Run: `pytest tests/test_dataset.py -v`

Expected: failures because all current scaffold positions are always masked.

- [ ] **Step 3: Implement mixed corruption without target leakage**

Build the full target canvas, choose full/random/span corruption using the per-item generator, replace selected scaffold tokens with `<MASK>`, keep unselected context visible, and add `prediction_mask`. Compute base loss and token accuracy on `prediction_mask`, not every non-motif position.

- [ ] **Step 4: Write scheduler and callback tests**

Use a fake trainer with known `estimated_stepping_batches`. Assert warmup reaches the base LR, cosine reaches `0.02 * lr`, scheduler interval is `step`, `ModelCheckpoint` saves best plus last, `EarlyStopping` monitors `val/loss`, and `--resume` reaches `trainer.fit(ckpt_path=...)`.

- [ ] **Step 5: Run schedule tests and verify RED**

Run: `pytest tests/test_training_schedule.py tests/test_training_config.py -v`

Expected: failures because the current scheduler uses fixed `T_max=1000` and no early stopping/resume.

- [ ] **Step 6: Implement warmup-cosine, label smoothing, callbacks, and resume**

Create a `LambdaLR` after trainer attachment using estimated optimizer steps, return interval `step`, apply label smoothing only to scaffold CE, add `EarlyStopping`, set checkpoint filename without metric path separators, and accept CLI `--resume` overriding config.

- [ ] **Step 7: Run training regressions**

Run: `pytest tests/test_dataset.py tests/test_denoising_model.py tests/test_training_schedule.py tests/test_training_config.py -v`

Expected: PASS.

### Task 4: Add RNAfold validation and transparent candidate ranking

**Files:**
- Create: `rna_scaffold/validators/__init__.py`
- Create: `rna_scaffold/validators/rnafold.py`
- Create: `rna_scaffold/ranking.py`
- Create: `validate_scaffolds.py`
- Test: `tests/test_rnafold_validator.py`
- Test: `tests/test_ranking.py`

**Interfaces:**
- Produces: `RnafoldResult(status, dot_bracket, mfe_kcal_mol, paired_fraction, motif_paired_fraction, runtime_seconds, version, error)`.
- Produces: `run_rnafold(sequence, motif_start, motif_end, executable="RNAfold", timeout_seconds=30) -> RnafoldResult`.
- Produces: `rank_candidates(candidates, rnafold_results=None, weights=None) -> list[RankedCandidate]` with every raw and normalized component retained.

- [ ] **Step 1: Write parser and failure tests**

Test canonical RNAfold output, whitespace variants, missing executable, timeout, non-zero exit, malformed output, paired fraction, and motif-local paired fraction using temporary fake executables.

- [ ] **Step 2: Run validator tests and verify RED**

Run: `pytest tests/test_rnafold_validator.py -v`

Expected: missing-module collection failure.

- [ ] **Step 3: Implement the subprocess adapter**

Call `[executable, "--noPS"]` with FASTA on stdin, `text=True`, `capture_output=True`, timeout, and no shell. Query version separately and return explicit statuses `ok`, `unavailable`, `timeout`, `failed`, or `parse_error`.

- [ ] **Step 4: Write ranking tests**

Assert deterministic ordering, retained raw components, no MFE-only preference, neutral handling of unavailable RNAfold, penalties for extreme GC/homopolymers/low entropy, and no mutation of candidate records.

- [ ] **Step 5: Implement ranking and validation CLI**

Normalize components within the candidate batch with constant-component protection. Default weights combine likelihood, entropy, GC reasonableness, homopolymer penalty, paired fraction, motif accessibility, and moderate length-normalized MFE. `validate_scaffolds.py` reads JSONL, validates all rows, ranks them, and atomically writes augmented JSONL.

- [ ] **Step 6: Run validator/ranking tests**

Run: `pytest tests/test_rnafold_validator.py tests/test_ranking.py -v`

Expected: PASS.

### Task 5: Add baselines, ablations, and statistical benchmark reports

**Files:**
- Create: `rna_scaffold/evaluation.py`
- Create: `benchmark_scaffolds.py`
- Create: `configs/benchmark_scaffolds.yaml`
- Create: `tests/test_scaffold_evaluation.py`
- Create: `tests/test_benchmark_cli.py`

**Interfaces:**
- Produces: `uniform_baseline`, `markov_baseline`, and checkpoint generator runners behind `GeneratorProtocol`.
- Produces: `paired_bootstrap(left, right, seed=42, samples=10000) -> BootstrapDifference`.
- Produces: candidate CSV, motif CSV, summary JSON, and run-manifest JSON.

- [ ] **Step 1: Write metrics and bootstrap tests**

Assert validity, motif preservation, unique rate, pairwise normalized edit diversity, k-mer nearest-training similarity, length error, failure counts, deterministic bootstrap intervals, and paired comparison by motif rather than candidate.

- [ ] **Step 2: Run evaluation tests and verify RED**

Run: `pytest tests/test_scaffold_evaluation.py -v`

Expected: missing-module collection failure.

- [ ] **Step 3: Implement dependency-light evaluation**

Use Python/NumPy only. Keep missing RNAfold values as null plus explicit counts. Hash configs, manifests, checkpoints, source input, and output artifacts. Record dependency versions, command arguments, wall time, seed, and candidate budget.

- [ ] **Step 4: Implement benchmark configuration and CLI**

The YAML declares held-out motif source, equal candidate count, seeds, training CSV for novelty/Markov statistics, method checkpoints, RNAfold policy, and output directory. `--smoke-test` creates a tiny deterministic workload covering uniform and Markov without external tools; learned methods require declared checkpoints.

- [ ] **Step 5: Run benchmark tests and smoke command**

Run: `pytest tests/test_scaffold_evaluation.py tests/test_benchmark_cli.py -v && python benchmark_scaffolds.py --config configs/benchmark_scaffolds.yaml --smoke-test`

Expected: PASS and four non-empty report artifacts with matching run hashes.

### Task 6: Documentation and final verification audit

**Files:**
- Modify: `README.md`
- Modify: `pyproject.toml`
- Create: `docs/scaffold_model_card.md`
- Create: `scripts/verify_scaffold_release.py`
- Test: `tests/test_package_boundary.py`
- Test: `tests/test_release_audit.py`

**Interfaces:**
- Adds console scripts for generation, validation, and benchmarking.
- Produces: `outputs/scaffold_release_audit.json` from deterministic local checks.

- [ ] **Step 1: Write documentation/interface boundary tests**

Assert package entry points resolve, README formal examples require checkpoint, Markov is labeled baseline, external validators are labeled optional, and no claim says local/server full training has completed without an artifact hash.

- [ ] **Step 2: Run boundary tests and verify RED**

Run: `pytest tests/test_package_boundary.py tests/test_release_audit.py -v`

Expected: failures for missing entry points and audit script.

- [ ] **Step 3: Update commands and model card**

Document architecture ownership, 1815-sequence MMseqs80 split preparation, checkpoint generation, RNAfold installation/status behavior, benchmark fairness, ablations, limitations, and exact server training/resume commands. Do not commit server-generated datasets or weights.

- [ ] **Step 4: Implement the release verifier**

Run the test suite and CLI help/smoke commands as explicit subprocess argument lists. Record command, return code, stdout/stderr tail, timestamp, Git commit, dirty paths, Python/package versions, and artifact SHA-256. Mark server-only training and RNAfold checks `not_run` when unavailable.

- [ ] **Step 5: Run complete verification**

Run: `pytest -q && python generate_scaffold.py --help && python validate_scaffolds.py --help && python benchmark_scaffolds.py --help && python scripts/verify_scaffold_release.py --output outputs/scaffold_release_audit.json`

Expected: all local tests and help commands pass; audit truthfully distinguishes passed, unavailable, and server-required checks.

- [ ] **Step 6: Inspect scope before handoff**

Run: `git status --short && git diff --check && git diff --stat`

Expected: only planned one-dimensional files plus the pre-existing user-owned `rna_scaffold/__init__.py` and `train_3d.py` state; no legacy checkpoint or 3D deletion.
