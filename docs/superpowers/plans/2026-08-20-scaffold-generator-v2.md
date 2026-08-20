# RNA Scaffold Generator V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace V1 with a clean V2 RNA scaffold generator using canonical homology-safe data, bilateral flank prediction, RNA-FM LoRA, confidence remasking, diversity controls, and a reproducible expanded benchmark.

**Architecture:** Keep the bidirectional denoising Transformer and stable command names, but replace the placement contract throughout the dataset, model, loss, checkpoints, generation, and benchmark. Keep external RNA-FM and RNAfold behind explicit adapters so unit tests remain CPU-only and deterministic.

**Tech Stack:** Python 3.10+, PyTorch, Lightning, RNA-FM, PyYAML, NumPy, ViennaRNA RNAfold subprocess, pytest.

**Spec:** `docs/superpowers/specs/2026-08-20-scaffold-generator-v2-design.md`

## Global Constraints

- V2 deliberately rejects V1 and unversioned checkpoints.
- Maximum model and generated sequence length is 512 nucleotides.
- Immutable motif bases may never be modified or remasked.
- Formal checkpoint methods may never fall back to heuristic generation.
- Unit tests may not download RNA-FM weights or require RNAfold.
- RNAfold metrics are plausibility proxies, not evidence of biological function.
- All behavior changes use red-green-refactor TDD.

---

### Task 1: Canonical cluster-manifest data contract

**Files:**
- Modify: `rna_scaffold/splits.py`
- Modify: `rna_scaffold/datamodule.py`
- Modify: `rna_scaffold/data.py`
- Modify: `configs/train_scaffold_a800.yaml`
- Modify: `tests/test_records_and_splits.py`
- Modify: `tests/test_dataset.py`
- Modify: `tests/test_training_config.py`

**Interfaces:**
- Produces: `load_cluster_assignments(path: str | Path) -> dict[str, str]`
- Produces: `build_cluster_disjoint_manifest(records, cluster_by_id, seed=42, val_fraction=..., test_fraction=...) -> SplitManifest`
- Produces batch keys `target_left_length` and `target_right_length`.
- Removes batch keys `target_length` and `motif_start` from the training contract after deriving bilateral targets.

- [ ] **Step 1: Write failing split tests**

Add tests proving a TSV assignment creates family-disjoint partitions, missing assignments fail, duplicate assignments fail, and a cluster appearing in two requested partitions fails. Use temporary records with two IDs in one cluster and assert both IDs share a partition.

- [ ] **Step 2: Run the split tests and verify RED**

Run: `python -m pytest tests/test_records_and_splits.py -q`

Expected: FAIL because cluster assignment loading and cluster-backed manifest construction do not exist.

- [ ] **Step 3: Implement strict cluster loading and splitting**

Parse the header fields `cluster_id`, `sequence_id`, and `cluster_size`; reject blank, duplicate, unknown, and incomplete assignments. Reuse `SplitManifest`, but assign complete clusters deterministically using the seed and target split fractions.

- [ ] **Step 4: Write failing dataset-target tests**

For a target `AACCGGUU` with motif positions `[2, 6)`, assert:

```python
assert item["target_left_length"].item() == 2
assert item["target_right_length"].item() == 2
assert "target_length" not in item
assert "motif_start" not in item
```

- [ ] **Step 5: Run dataset tests and verify RED**

Run: `python -m pytest tests/test_dataset.py -q`

Expected: FAIL because V1 target keys are still returned.

- [ ] **Step 6: Implement V2 bilateral targets and data-module wiring**

Add a required `cluster_manifest` argument for formal training data, pass assignments into dataset partition construction, and retain dynamic cropping for sequences longer than 512.

- [ ] **Step 7: Update and test the A800 configuration**

Set repository-relative canonical paths:

```yaml
data:
  train_data: data/processed/rna_merged_mmseqs80.csv
  cluster_manifest: outputs/mmseqs80_search/rna80_connected_components.tsv
```

Assert the configuration has no reference to `train_sequences.csv` and includes both canonical files.

- [ ] **Step 8: Verify Task 1 GREEN and commit**

Run: `python -m pytest tests/test_records_and_splits.py tests/test_dataset.py tests/test_training_config.py -q`

Commit: `feat: enforce canonical clustered scaffold data`

---

### Task 2: Bilateral flank model and losses

**Files:**
- Modify: `rna_scaffold/model.py`
- Modify: `rna_scaffold/lightning_module.py`
- Modify: `tests/test_denoising_model.py`
- Modify: `tests/test_training_schedule.py`

**Interfaces:**
- Produces `ScaffoldModelOutput(token_logits, left_length_logits, right_length_logits)`.
- Produces `DenoisingLosses(total_loss, base_loss, left_length_loss, right_length_loss)`.
- `compute_denoising_losses(..., target_left_length, target_right_length, left_length_loss_weight, right_length_loss_weight)`.

- [ ] **Step 1: Replace V1 output assertions with failing V2 assertions**

Assert both flank logits have shape `[batch, max_length + 1]` and assert `length_logits` and `position_logits` are absent.

- [ ] **Step 2: Run the model test and verify RED**

Run: `python -m pytest tests/test_denoising_model.py::test_model_outputs_tokens_and_bilateral_flank_lengths -q`

Expected: FAIL because V1 fields still exist.

- [ ] **Step 3: Implement V2 output heads**

Replace `length_head` and `position_head` with independent `left_length_head` and `right_length_head`, each covering classes `0..max_length`.

- [ ] **Step 4: Write failing bilateral-loss tests**

Construct logits whose left target is 2 and right target is 3. Verify each auxiliary loss changes only when its corresponding target changes, while fixed motif targets do not affect base loss.

- [ ] **Step 5: Run loss tests and verify RED**

Run: `python -m pytest tests/test_denoising_model.py -q`

Expected: FAIL because loss still accepts V1 targets.

- [ ] **Step 6: Implement V2 losses and Lightning step**

Keep compact motif-only placement inference. Log `left_length_loss`, `right_length_loss`, `left_length_accuracy`, and `right_length_accuracy`. Remove V1 hyperparameters and logs.

- [ ] **Step 7: Verify Task 2 GREEN and commit**

Run: `python -m pytest tests/test_denoising_model.py tests/test_training_schedule.py -q`

Commit: `feat: predict bilateral scaffold lengths`

---

### Task 3: V2 checkpoint boundary

**Files:**
- Modify: `rna_scaffold/checkpoints.py`
- Modify: `rna_scaffold/lightning_module.py`
- Modify: `tests/test_checkpoint_loading.py`
- Modify: `tests/test_release_audit.py`

**Interfaces:**
- Checkpoints contain top-level `architecture_version: 2`.
- `load_scaffold_checkpoint` rejects every version other than integer `2`.

- [ ] **Step 1: Write failing version tests**

Create one tiny V2 checkpoint and two invalid checkpoints: missing version and version 1. Assert invalid loads raise `CheckpointCompatibilityError` matching `architecture_version 2`.

- [ ] **Step 2: Run checkpoint tests and verify RED**

Run: `python -m pytest tests/test_checkpoint_loading.py -q`

Expected: FAIL because version metadata is not enforced.

- [ ] **Step 3: Implement save/load version enforcement**

Write the version in `on_save_checkpoint`, validate before constructing the Lightning module, and report retraining as the remedy. Do not translate V1 hyperparameter names.

- [ ] **Step 4: Verify Task 3 GREEN and commit**

Run: `python -m pytest tests/test_checkpoint_loading.py tests/test_release_audit.py -q`

Commit: `feat: enforce scaffold checkpoint v2`

---

### Task 4: RNA-FM none, frozen, and LoRA modes

**Files:**
- Modify: `rna_scaffold/pretrained.py`
- Modify: `rna_scaffold/lightning_module.py`
- Modify: `configs/train_scaffold_a800.yaml`
- Modify: `requirements.txt`
- Modify: `tests/test_pretrained_adapter.py`
- Modify: `tests/test_training_config.py`

**Interfaces:**
- `PretrainedEncoderConfig(kind: str = "none", checkpoint: str | None = None, mode: str = "frozen", lora_rank: int = 8, lora_alpha: float = 16.0, lora_dropout: float = 0.05, lora_last_n_layers: int = 4)`.
- `inject_lora_adapters(model, last_n_layers, rank, alpha, dropout) -> tuple[str, ...]`.
- `LoRALinear` preserves the wrapped frozen linear output plus a trainable low-rank residual.

- [ ] **Step 1: Write failing LoRA unit tests using a tiny RNA-FM-shaped module**

Assert frozen base weights have `requires_grad=False`, adapter A/B weights have `requires_grad=True`, non-target early layers have no adapters, zero-initialized adapters preserve the original output, and unsupported layouts raise a diagnostic error.

- [ ] **Step 2: Run adapter tests and verify RED**

Run: `python -m pytest tests/test_pretrained_adapter.py -q`

Expected: FAIL because LoRA types and injection do not exist.

- [ ] **Step 3: Implement dependency-free LoRA wrappers**

Implement LoRA with native `torch.nn` to avoid a new PEFT dependency. Discover supported attention projections only inside the final configured layer count; freeze the RNA-FM base before injection and record exact injected names.

- [ ] **Step 4: Write failing configuration and metadata tests**

Assert `none`, `frozen`, and `lora` construct correctly; reject V1 `freeze`; verify rank, alpha, dropout, layer count, target names, checkpoint hash, and trainable counts appear in checkpoint metadata.

- [ ] **Step 5: Implement mode construction and metadata**

Keep frozen RNA-FM in evaluation mode. Keep LoRA RNA-FM base deterministic while adapter dropout follows module training state. Ensure optimizer receives adapters, project Transformer, projection, and heads.

- [ ] **Step 6: Update A800 defaults**

Use LoRA rank 8, alpha 16, dropout 0.05, and last four layers. Keep `rna-fm` as the server dependency and document its cached checkpoint behavior.

- [ ] **Step 7: Verify Task 4 GREEN and commit**

Run: `python -m pytest tests/test_pretrained_adapter.py tests/test_training_config.py tests/test_checkpoint_loading.py -q`

Commit: `feat: add RNA-FM LoRA adaptation`

---

### Task 5: Joint legal flank selection

**Files:**
- Modify: `rna_scaffold/decoding.py`
- Modify: `rna_scaffold/generate.py`
- Modify: `tests/test_iterative_decoding.py`
- Modify: `tests/test_generate.py`

**Interfaces:**
- Replaces `select_length_position` with `select_flank_lengths(output, motif_length, max_length, generator, sample, min_scaffold_length, min_flank_length) -> tuple[int, int]`.

- [ ] **Step 1: Write failing legal-pair tests**

Give maximal marginal logits to an illegal pair and the second-best logits to a legal pair. Assert deterministic selection returns the legal pair and sampled selection never returns an illegal pair across fixed seeds.

- [ ] **Step 2: Run decoding tests and verify RED**

Run: `python -m pytest tests/test_iterative_decoding.py -q`

Expected: FAIL because V2 selection does not exist.

- [ ] **Step 3: Implement masked joint scoring**

Enumerate only legal `(left, right)` pairs, sum log probabilities, then sample or argmax. Return bilateral lengths and derive `total_length` and `motif_start` only at canvas construction.

- [ ] **Step 4: Migrate generation and verify GREEN**

Run: `python -m pytest tests/test_iterative_decoding.py tests/test_generate.py -q`

Commit: `feat: select valid bilateral scaffold lengths`

---

### Task 6: Confidence remasking decoder

**Files:**
- Modify: `rna_scaffold/decoding.py`
- Modify: `tests/test_iterative_decoding.py`

**Interfaces:**
- `DecodingSettings` adds a deterministic mask schedule selector.
- `DecodedScaffold` reports `masked_counts` and final normalized log probability.

- [ ] **Step 1: Write failing remasking behavior tests**

Use a stateful deterministic model whose preferred base changes after context is visible. Assert a low-confidence previously filled position is remasked and corrected, mask counts decrease monotonically to zero, final sequence contains only RNA bases, and motif tokens are identical after every forward pass.

- [ ] **Step 2: Run the targeted test and verify RED**

Run: `python -m pytest tests/test_iterative_decoding.py -q`

Expected: FAIL because V1 only commits unresolved positions once.

- [ ] **Step 3: Implement confidence remasking**

At every pass, sample all scaffold positions, calculate confidence, retain the highest-confidence count required by the schedule, remask the remainder, restore motif tokens, and force zero masks on the final pass. Re-run the final sequence once to score its actual tokens.

- [ ] **Step 4: Verify deterministic and stochastic behavior GREEN**

Run: `python -m pytest tests/test_iterative_decoding.py -q`

Commit: `feat: add confidence remasking decoder`

---

### Task 7: Candidate diversity controls and accounting

**Files:**
- Modify: `rna_scaffold/generate.py`
- Modify: `generate_scaffold.py`
- Modify: `tests/test_generate.py`
- Modify: `tests/test_generate_cli.py`

**Interfaces:**
- `GenerationSettings` adds `max_attempts`, `min_normalized_edit_distance`, optional `max_kmer_similarity`, `kmer_size`, optional `max_homopolymer_run`, and optional `gc_min/gc_max` soft ranking bounds.
- Generation returns candidates plus an audit containing requested, attempted, accepted, and per-reason rejection counts.

- [ ] **Step 1: Write failing diversity tests**

Feed duplicate, near-duplicate, excessive-homopolymer, and acceptable sequences through the acceptance unit. Assert exact rejection reasons and that no accepted sequence is duplicated to meet a quota.

- [ ] **Step 2: Run generation tests and verify RED**

Run: `python -m pytest tests/test_generate.py tests/test_generate_cli.py -q`

Expected: FAIL because diversity policy and audit do not exist.

- [ ] **Step 3: Implement a focused diversity policy and bounded loop**

Keep filtering pure and separately testable. Treat GC bounds as ranking metadata rather than a hard rejection unless explicitly configured. Diversify flank sampling by retaining sampled legal pairs across attempts.

- [ ] **Step 4: Update atomic JSONL/FASTA output metadata**

Include architecture version, generation settings, and audit. Preserve atomic file replacement and return a non-zero CLI status only for invalid inputs or checkpoint failures, not for a short but valid candidate set.

- [ ] **Step 5: Verify Task 7 GREEN and commit**

Run: `python -m pytest tests/test_generate.py tests/test_generate_cli.py -q`

Commit: `feat: enforce scaffold candidate diversity`

---

### Task 8: Benchmark V2 methods and metrics

**Files:**
- Modify: `rna_scaffold/generate.py`
- Modify: `rna_scaffold/evaluation.py`
- Modify: `benchmark_scaffolds.py`
- Modify: `configs/benchmark_scaffolds.yaml`
- Modify: `tests/test_scaffold_evaluation.py`
- Modify: `tests/test_benchmark_cli.py`

**Interfaces:**
- `RnaTrainingPrior` supports explicit first- and second-order sampling.
- `CandidateMetric` adds edit diversity inputs, nearest-training similarity, homopolymer run, elapsed time, optional CUDA peak memory, and optional RNAfold fields.
- Benchmark writes the six artifact names required by the spec.

- [ ] **Step 1: Write failing second-order Markov tests**

Use a corpus with deterministic two-base contexts and prove second-order output follows the two-base transition while first-order output cannot represent that distinction.

- [ ] **Step 2: Run baseline tests and verify RED**

Run: `python -m pytest tests/test_scaffold_evaluation.py tests/test_benchmark_cli.py -q`

Expected: FAIL because only one Markov mode and the V1 artifacts exist.

- [ ] **Step 3: Implement explicit uniform, Markov-1, and Markov-2 methods**

Use equal motif and candidate budgets, deterministic per-method seeds, and no heuristic fallback for checkpoint methods.

- [ ] **Step 4: Write failing metric and artifact tests**

Assert maximum homopolymer run, within-motif edit diversity, nearest-training 5-mer similarity, failure counts, paired bootstrap reproducibility, and exact artifact filenames. Assert the configured primary model is marked in summaries and reports.

- [ ] **Step 5: Implement expanded metrics and optional RNAfold enrichment**

Reuse `rna_scaffold.validators.rnafold` statuses. Aggregate candidate rows by motif and method, compute paired bootstrap only on common motif IDs, capture CUDA memory only on CUDA, and write artifacts atomically.

- [ ] **Step 6: Implement manifest and Markdown report**

Record config/data/checkpoint hashes, seed, command, software versions, method status, runtime, and artifact hashes. Clearly label the LoRA iterative method as `primary: true` and RNAfold results as plausibility proxies.

- [ ] **Step 7: Verify Task 8 GREEN and commit**

Run: `python -m pytest tests/test_scaffold_evaluation.py tests/test_benchmark_cli.py tests/test_rnafold_validator.py -q`

Commit: `feat: expand reproducible scaffold benchmark`

---

### Task 9: Documentation, release audit, and end-to-end verification

**Files:**
- Modify: `README.md`
- Modify: `docs/scaffold_model_card.md`
- Modify: `docs/IMPROVEMENT_AUDIT.md`
- Modify: `scripts/verify_scaffold_release.py`
- Modify: `scripts/setup_server_env.sh`
- Modify: `tests/test_package_boundary.py`
- Modify: `tests/test_release_audit.py`

**Interfaces:**
- Documents one authoritative V2 server workflow and formal benchmark workflow.
- Release audit verifies no V1 placement field, V1 checkpoint path, or removed 3D training module is referenced by runtime code/configuration.

- [ ] **Step 1: Write failing release-boundary tests**

Assert runtime Python and YAML files contain none of `position_logits`, `length_logits`, `position_loss_weight`, `length_loss_weight`, `train_3d`, or V1 fallback wording, except migration explanations in documentation.

- [ ] **Step 2: Run release tests and verify RED**

Run: `python -m pytest tests/test_package_boundary.py tests/test_release_audit.py -q`

Expected: FAIL until all V1 runtime references and configuration fields are removed.

- [ ] **Step 3: Update documentation and server setup**

Document canonical data checks, RNA-FM cache download, LoRA training configuration, V2 checkpoint rejection, smoke training, full A800 launch, generation, RNAfold validation, benchmark, and resume commands.

- [ ] **Step 4: Run CPU smoke training and generation**

Run the existing tiny CPU training fixture to create a V2 checkpoint, load it, generate at least two motif-preserving candidates, and evaluate them without RNAfold.

- [ ] **Step 5: Run the complete verification suite**

Run:

```text
python -m pytest -q
python scripts/verify_scaffold_release.py
python benchmark_scaffolds.py --config configs/benchmark_scaffolds.yaml --output-dir outputs/v2_smoke --smoke-test
git diff --check
```

Expected: all tests pass; release audit passes; smoke benchmark creates all six V2 artifacts; no whitespace errors.

- [ ] **Step 6: Review final diff and commit**

Confirm only V2-scoped files changed and no generated data, checkpoints, RNA-FM weights, or benchmark outputs are staged.

Commit: `docs: finalize scaffold generator v2 workflow`
