# RNA Scaffold Generator Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the placeholder Markov scaffold path with a checkpoint-backed, motif-protected, variable-length RNA generator and remove the project-owned 3D subsystem.

**Architecture:** A leakage-safe dataset samples variable motifs from family-disjoint RNA sequences. A bidirectional denoising Transformer predicts a valid total length, motif offset, and both scaffold flanks while immutable motif tokens are restored after every decoding step. External RNAfold and official RhoFold+ are isolated subprocess adapters used only for validation and ranking.

**Tech Stack:** Python 3.10+, PyTorch 2.1+, Lightning 2.3+, pytest 8+, YAML, optional ViennaRNA CLI and official RhoFold+ checkout.

**Spec:** `docs/superpowers/specs/2026-08-15-scaffold-generator-redesign.md`

## Global Constraints

- Accepted motif alphabet is exactly A/U/C/G and motif length is 4–64 nucleotides.
- Generated total length is automatic, strictly greater than motif length, and no greater than 512 nucleotides.
- Motif tokens are immutable during training and generation.
- Dataset partitions are family-disjoint and fail closed on exact-sequence overlap.
- External validators are optional adapters and cannot be imported by the core generator.
- Every stochastic interface accepts a seed and must reproduce results on the same software/hardware stack.
- Existing user data and one-dimensional checkpoints are retained; local 3D checkpoints are deleted only as explicitly approved.

---

### Task 1: Remove the project-owned 3D subsystem

**Files:**
- Delete: `rna_scaffold_3d/`
- Delete: `train_3d.py`
- Delete: `fold_3d.py`
- Delete: `evaluate_3d.py`
- Delete: `configs/train_3d_a800_card1.yaml`
- Delete: `configs/train_3d_a800_full.yaml`
- Delete: `configs/train_3d_local_windows.yaml`
- Delete: `scripts/train_3d_a800_card1.sh`
- Delete: `scripts/train_and_evaluate_3d.py`
- Delete: `scripts/train_and_evaluate_3d.sh`
- Delete: all `tests/test_*3d*.py`, `tests/test_fold_3d.py`, and `tests/test_pdb_data.py`
- Delete: `checkpoints_3d/`
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Modify: `README.md`
- Test: `tests/test_package_boundary.py`

**Interfaces:**
- Consumes: current package metadata and repository paths.
- Produces: a one-dimensional-only Python package with console script `rna-generate-scaffold = "generate_scaffold:main"` and no importable `rna_scaffold_3d` package.

- [ ] **Step 1: Write the failing package-boundary test**

```python
from pathlib import Path


def test_local_3d_subsystem_is_not_shipped():
    root = Path(__file__).resolve().parents[1]
    forbidden = [
        root / "rna_scaffold_3d",
        root / "train_3d.py",
        root / "fold_3d.py",
        root / "evaluate_3d.py",
    ]
    assert not [str(path) for path in forbidden if path.exists()]
```

- [ ] **Step 2: Run the boundary test and verify it fails**

Run: `pytest tests/test_package_boundary.py -v`

Expected: FAIL listing the existing local 3D files.

- [ ] **Step 3: Delete only the approved 3D paths and update metadata**

Set `pyproject.toml` to describe a motif-conditioned scaffold generator, retain only `train` and `generate_scaffold` in `py-modules`, remove the three local-3D console scripts, and restrict package discovery to `rna_scaffold*`. Remove `gemmi` when no remaining one-dimensional import requires it. Rewrite README architecture and commands around sequence generation plus optional external validators.

- [ ] **Step 4: Run the boundary and existing one-dimensional tests**

Run: `pytest tests/test_package_boundary.py tests/test_rna_utils.py tests/test_dataset.py tests/test_model.py tests/test_generate.py -v`

Expected: PASS with no imports from deleted modules.

- [ ] **Step 5: Commit the removal**

```bash
git add -A -- rna_scaffold_3d train_3d.py fold_3d.py evaluate_3d.py configs scripts tests pyproject.toml requirements.txt README.md checkpoints_3d
git commit -m "refactor: remove local RNA 3D subsystem"
```

### Task 2: Add normalized records, family-disjoint splits, and variable motif sampling

**Files:**
- Create: `rna_scaffold/records.py`
- Create: `rna_scaffold/splits.py`
- Modify: `rna_scaffold/data.py`
- Modify: `rna_scaffold/datamodule.py`
- Test: `tests/test_records_and_splits.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: CSV rows with `sequence` and optional `target_id`, `family`, `source`, and `release_date`.
- Produces: `RnaSequenceRecord`, `SplitManifest`, `MotifScaffoldExample`, `build_family_disjoint_manifest(records, seed)`, and `sample_motif_example(record, generator, min_motif_length=4, max_motif_length=64, max_total_length=512)`.

- [ ] **Step 1: Write failing normalization and leakage tests**

```python
def test_family_members_never_cross_partitions():
    records = [
        RnaSequenceRecord("a", "AUGCAUGC", "RF1", "test"),
        RnaSequenceRecord("b", "AUGCAUGG", "RF1", "test"),
        RnaSequenceRecord("c", "CCCCAAAA", "RF2", "test"),
    ]
    manifest = build_family_disjoint_manifest(records, seed=7)
    assert manifest.partition_for("a") == manifest.partition_for("b")


def test_manifest_rejects_exact_sequence_overlap():
    records = [
        RnaSequenceRecord("a", "AUGCAUGC", "RF1", "test"),
        RnaSequenceRecord("b", "AUGCAUGC", "RF2", "test"),
    ]
    with pytest.raises(ValueError, match="exact sequence overlap"):
        validate_manifest(records, {"train": ["a"], "test": ["b"]})
```

- [ ] **Step 2: Run the new tests and verify missing symbols fail**

Run: `pytest tests/test_records_and_splits.py -v`

Expected: collection ERROR for missing `rna_scaffold.records` or `rna_scaffold.splits`.

- [ ] **Step 3: Implement immutable records and split manifests**

```python
@dataclass(frozen=True)
class RnaSequenceRecord:
    target_id: str
    sequence: str
    family: str | None
    source: str

    def __post_init__(self) -> None:
        normalized = self.sequence.strip().upper().replace("T", "U")
        if not 2 <= len(normalized) <= 512 or set(normalized) - set("AUCG"):
            raise ValueError(f"invalid RNA sequence for {self.target_id}")
        object.__setattr__(self, "sequence", normalized)
```

Build partitions by grouped family key, use a stable hash plus seed for assignment, save record IDs and sequence hashes, and validate IDs, family membership, and exact sequences before use.

- [ ] **Step 4: Implement variable motif examples**

```python
@dataclass(frozen=True)
class MotifScaffoldExample:
    motif: str
    target_sequence: str
    motif_start: int

    @property
    def total_length(self) -> int:
        return len(self.target_sequence)
```

Use `torch.Generator` for reproducibility. Clamp motif length to `min(64, len(sequence) - 1)`, sample lengths from 4 upward, sample all valid offsets, and include an epoch-dependent seed in the dataset.

- [ ] **Step 5: Run dataset and split tests**

Run: `pytest tests/test_records_and_splits.py tests/test_dataset.py -v`

Expected: PASS including terminal and asymmetric motif placements.

- [ ] **Step 6: Commit the data boundary**

```bash
git add rna_scaffold/records.py rna_scaffold/splits.py rna_scaffold/data.py rna_scaffold/datamodule.py tests/test_records_and_splits.py tests/test_dataset.py
git commit -m "feat: add leakage-safe variable motif datasets"
```

### Task 3: Implement the motif-protected denoising generator

**Files:**
- Create: `rna_scaffold/model.py`
- Modify: `rna_scaffold/lightning_module.py`
- Modify: `rna_scaffold/tokenizer.py`
- Modify: `configs/train_stanford_1d.yaml`
- Test: `tests/test_denoising_model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: padded `input_ids`, `fixed_mask`, `attention_mask`, `target_ids`, `target_length`, and `motif_start` tensors.
- Produces: `ScaffoldModelOutput(token_logits, length_logits, position_logits, confidence)` and `MotifDenoisingTransformer.forward(...)`.

- [ ] **Step 1: Write failing shape and motif-mask tests**

```python
def test_model_outputs_tokens_lengths_and_positions():
    model = MotifDenoisingTransformer(vocab_size=12, d_model=32, nhead=4, num_layers=2, max_length=64)
    output = model(
        input_ids=torch.tensor([[3, 3, 8, 11, 3, 3]]),
        attention_mask=torch.ones(1, 6, dtype=torch.bool),
    )
    assert output.token_logits.shape == (1, 6, 4)
    assert output.length_logits.shape == (1, 65)
    assert output.position_logits.shape == (1, 64)


def test_restore_fixed_tokens_never_changes_motif():
    original = torch.tensor([[3, 8, 11, 3]])
    proposed = torch.tensor([[9, 9, 9, 9]])
    fixed = torch.tensor([[False, True, True, False]])
    assert restore_fixed_tokens(proposed, original, fixed).tolist() == [[9, 8, 11, 9]]
```

- [ ] **Step 2: Run tests and verify missing implementation fails**

Run: `pytest tests/test_denoising_model.py -v`

Expected: collection ERROR for missing `rna_scaffold.model`.

- [ ] **Step 3: Implement the focused model module**

Use batch-first `nn.TransformerEncoder`, learned absolute positions through index 511, a four-base scaffold head, a 513-class length head with invalid classes masked, a 512-class position head, and a scalar confidence head. Keep Lightning-specific loss and logging outside this module.

- [ ] **Step 4: Replace autoregressive Lightning training with denoising losses**

`RnaScaffoldLitModule._step` computes base cross-entropy only on non-fixed valid positions plus weighted length and position losses. It logs `base_loss`, `length_loss`, `position_loss`, motif-preservation assertions, token accuracy, and total loss. Configuration exposes each loss weight and `max_length: 512`.

- [ ] **Step 5: Run tiny-overfit and unit tests**

Run: `pytest tests/test_denoising_model.py tests/test_model.py -v`

Expected: PASS; the tiny synthetic batch loss decreases after 50 optimizer steps and motif tokens remain unchanged.

- [ ] **Step 6: Commit the model**

```bash
git add rna_scaffold/model.py rna_scaffold/lightning_module.py rna_scaffold/tokenizer.py configs/train_stanford_1d.yaml tests/test_denoising_model.py tests/test_model.py
git commit -m "feat: add motif-protected denoising scaffold model"
```

### Task 4: Implement checkpoint-backed iterative generation

**Files:**
- Rewrite: `rna_scaffold/generate.py`
- Create: `generate_scaffold.py`
- Test: `tests/test_generate.py`
- Test: `tests/test_generate_cli.py`

**Interfaces:**
- Consumes: `GenerationRequest(motif, num_candidates, max_length, seed, temperature, top_k, top_p, denoise_steps)` and a trained checkpoint.
- Produces: `ScaffoldCandidate` records and JSONL/FASTA output.

- [ ] **Step 1: Write failing public-API tests**

```python
def test_generate_candidates_preserves_variable_motif(fake_checkpoint):
    request = GenerationRequest(motif="AUGCGU", num_candidates=8, max_length=64, seed=13)
    candidates = generate_candidates(fake_checkpoint, request, device="cpu")
    assert len(candidates) == 8
    assert all(candidate.full_sequence[candidate.motif_start:candidate.motif_end] == request.motif for candidate in candidates)
    assert all(len(request.motif) < candidate.total_length <= 64 for candidate in candidates)


def test_generation_is_reproducible(fake_checkpoint):
    request = GenerationRequest(motif="GCGG", num_candidates=4, max_length=48, seed=5)
    assert generate_candidates(fake_checkpoint, request) == generate_candidates(fake_checkpoint, request)
```

- [ ] **Step 2: Run tests and verify the old Markov API fails**

Run: `pytest tests/test_generate.py tests/test_generate_cli.py -v`

Expected: FAIL because `GenerationRequest`, `ScaffoldCandidate`, and checkpoint-backed generation do not exist.

- [ ] **Step 3: Implement length/position sampling and iterative denoising**

At each step, sample allowed bases after temperature/top-k/top-p filtering, calculate confidence, lock a monotonically increasing fraction of the highest-confidence unresolved positions, and restore motif IDs unconditionally. Deduplicate by complete sequence while preserving generation attempts and stop with an explicit shortfall if the requested unique count cannot be reached.

- [ ] **Step 4: Implement atomic JSONL and FASTA output**

Write to a sibling `.tmp` file, flush and close it, then replace the final path. Each JSON row includes sequence, flanks, motif coordinates, sampling settings, normalized log probability, checkpoint SHA-256, and status.

- [ ] **Step 5: Run CLI smoke test**

Run: `python generate_scaffold.py --motif GCGG --checkpoint tests/fixtures/tiny_scaffold.ckpt --num-candidates 4 --max-length 48 --seed 42 --output .pytest_tmp/candidates.jsonl`

Expected: exit code 0, four valid JSON rows, and exact motif preservation.

- [ ] **Step 6: Commit the generation path**

```bash
git add rna_scaffold/generate.py generate_scaffold.py tests/test_generate.py tests/test_generate_cli.py
git commit -m "feat: generate ranked scaffolds from checkpoints"
```

### Task 5: Add optional RNA-FM initialization without coupling inference to it

**Files:**
- Create: `rna_scaffold/pretrained.py`
- Modify: `rna_scaffold/model.py`
- Modify: `train.py`
- Modify: `configs/train_stanford_1d.yaml`
- Test: `tests/test_pretrained_adapter.py`

**Interfaces:**
- Consumes: `PretrainedEncoderConfig(kind="none" | "rna_fm", checkpoint, freeze)`.
- Produces: `build_pretrained_encoder(config) -> nn.Module | None` and a projection into the generator `d_model`.

- [ ] **Step 1: Write failing frozen-encoder test**

```python
def test_frozen_pretrained_encoder_has_no_trainable_parameters(monkeypatch):
    monkeypatch.setattr(pretrained, "load_rna_fm", lambda _: TinyEncoder())
    encoder = build_pretrained_encoder(PretrainedEncoderConfig(kind="rna_fm", checkpoint="fake.pt", freeze=True))
    assert encoder is not None
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
```

- [ ] **Step 2: Run the adapter test and verify it fails**

Run: `pytest tests/test_pretrained_adapter.py -v`

Expected: collection ERROR for missing `rna_scaffold.pretrained`.

- [ ] **Step 3: Implement lazy optional loading**

Do not add RNA-FM to base dependencies. Import it only when `kind == "rna_fm"`; give a command containing the official repository/environment requirement when unavailable. Save the initializer kind, checkpoint hash, frozen state, and projection shape in every model checkpoint.

- [ ] **Step 4: Verify pure-project and frozen-RNA-FM configurations**

Run: `pytest tests/test_pretrained_adapter.py tests/test_denoising_model.py -v`

Expected: PASS with the mocked encoder and with `kind="none"` on a machine without RNA-FM.

- [ ] **Step 5: Commit the optional initializer**

```bash
git add rna_scaffold/pretrained.py rna_scaffold/model.py train.py configs/train_stanford_1d.yaml tests/test_pretrained_adapter.py
git commit -m "feat: add optional RNA-FM initialization"
```

### Task 6: Add external RNAfold and RhoFold+ validator adapters

**Files:**
- Create: `rna_scaffold/validators/__init__.py`
- Create: `rna_scaffold/validators/rnafold.py`
- Create: `rna_scaffold/validators/rhofold.py`
- Create: `validate_scaffolds.py`
- Test: `tests/test_external_validators.py`

**Interfaces:**
- Consumes: candidate JSONL, executable/repository paths, timeout, and top-k budget.
- Produces: `RnafoldResult`, `RhoFoldResult`, and augmented candidate JSONL without importing either external package.

- [ ] **Step 1: Write failing parser, timeout, and missing-tool tests**

```python
def test_parse_rnafold_output():
    result = parse_rnafold_output("AUGC\n(()) (-1.20)\n")
    assert result.dot_bracket == "(())"
    assert result.mfe_kcal_mol == pytest.approx(-1.2)


def test_missing_rhofold_is_recorded_not_fabricated(tmp_path):
    result = run_rhofold("AUGC", repository=tmp_path / "missing", timeout_seconds=1)
    assert result.status == "unavailable"
    assert result.pdb_path is None
```

- [ ] **Step 2: Run tests and verify missing adapters fail**

Run: `pytest tests/test_external_validators.py -v`

Expected: collection ERROR for missing validators.

- [ ] **Step 3: Implement isolated subprocess adapters**

RNAfold receives FASTA through standard input and parses dot-bracket/MFE output. RhoFold+ writes a temporary FASTA, invokes the official `inference.py --single_seq_pred True`, and collects `unrelaxed_model.pdb`, `results.npz`, `ss.ct`, log, runtime, exit code, and stderr. Neither adapter uses `shell=True`.

- [ ] **Step 4: Implement validation CLI and composite component fields**

The CLI validates all candidates with RNAfold, chooses a deterministic top-k using declared sequence metrics, optionally runs RhoFold+, and writes a new atomic result file. Raw component values remain separate; no undocumented scalar score is permitted.

- [ ] **Step 5: Run adapter and CLI tests**

Run: `pytest tests/test_external_validators.py -v`

Expected: PASS using fake executable scripts, including timeout and non-zero exit paths.

- [ ] **Step 6: Commit external validation**

```bash
git add rna_scaffold/validators validate_scaffolds.py tests/test_external_validators.py
git commit -m "feat: add external RNA structure validators"
```

### Task 7: Add baselines, ablations, and reproducible reporting

**Files:**
- Create: `rna_scaffold/evaluation.py`
- Create: `benchmark_scaffolds.py`
- Create: `configs/benchmark_scaffolds.yaml`
- Test: `tests/test_scaffold_evaluation.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: held-out motif manifest, method checkpoints/configurations, equal candidate budget, seeds, and optional structural validators.
- Produces: per-candidate CSV, per-motif CSV, summary JSON, bootstrap intervals, paired comparisons, and run manifest.

- [ ] **Step 1: Write failing metric and paired-bootstrap tests**

```python
def test_motif_preservation_metric_counts_every_candidate():
    rows = [CandidateMetric("m1", True, True), CandidateMetric("m1", False, False)]
    assert summarize_candidates(rows).motif_preservation_rate == pytest.approx(0.5)


def test_paired_bootstrap_is_reproducible():
    first = paired_bootstrap([1, 2, 3], [0, 1, 1], seed=42, samples=1000)
    second = paired_bootstrap([1, 2, 3], [0, 1, 1], seed=42, samples=1000)
    assert first == second
```

- [ ] **Step 2: Run evaluation tests and verify they fail**

Run: `pytest tests/test_scaffold_evaluation.py -v`

Expected: collection ERROR for missing `rna_scaffold.evaluation`.

- [ ] **Step 3: Implement baselines and raw metrics**

Implement uniform, first-order Markov, existing Encoder–Decoder, no-pretraining, no-structure-preference, and complete-model runners behind one `GeneratorProtocol`. Compute validity, motif preservation, unique rate, pairwise edit diversity, k-mer nearest-training similarity, length calibration, runtime, memory, and external-validator fields.

- [ ] **Step 4: Implement paired summaries and manifest hashing**

Use motif as the paired unit, fixed bootstrap seeds, percentile 95% intervals, and explicit missingness counts. Hash test manifest, generator checkpoint, config, dependency versions, and external-tool versions into the run manifest.

- [ ] **Step 5: Run the complete local test suite**

Run: `pytest -q`

Expected: PASS with all local-3D tests absent and all external tools mocked.

- [ ] **Step 6: Run a tiny end-to-end benchmark**

Run: `python benchmark_scaffolds.py --config configs/benchmark_scaffolds.yaml --smoke-test`

Expected: summary JSON and per-candidate/per-motif CSV files covering at least uniform, Markov, and tiny learned-model methods with identical budgets.

- [ ] **Step 7: Update documentation and commit**

```bash
git add rna_scaffold/evaluation.py benchmark_scaffolds.py configs/benchmark_scaffolds.yaml tests/test_scaffold_evaluation.py README.md
git commit -m "feat: benchmark scaffold generation rigorously"
```

### Task 8: Final verification and release audit

**Files:**
- Modify: `README.md`
- Create: `docs/scaffold_model_card.md`
- Create: `outputs/scaffold_release_audit.json`

**Interfaces:**
- Consumes: finished code, tests, smoke benchmark, checkpoint metadata, and split manifest.
- Produces: an evidence-backed release audit that distinguishes implemented, locally verified, server-required, and unavailable results.

- [ ] **Step 1: Run static repository boundary checks**

Run: `rg -n "rna_scaffold_3d|train_3d|fold_3d|evaluate_3d|local RhoFold" --glob '!docs/superpowers/**' .`

Expected: no product, package, configuration, or README reference to the deleted local 3D subsystem.

- [ ] **Step 2: Run tests and command help smoke checks**

Run: `pytest -q && python generate_scaffold.py --help && python validate_scaffolds.py --help && python benchmark_scaffolds.py --help`

Expected: every command exits zero and the test suite passes.

- [ ] **Step 3: Verify checkpoint-backed generation**

Run the tiny fixture twice with the same seed and once with a different seed. Assert identical outputs for equal seeds, at least one changed scaffold for the different seed, 100% motif preservation, and all lengths in `(motif_length, 512]`.

- [ ] **Step 4: Write the model card and audit artifact**

Document data sources, exclusions, family split, model ownership, RNA-FM attribution, intended use, limitations, structural-validator versions, benchmark budgets, and known failure modes. The JSON audit stores each verification command, exit code, timestamp, and output-artifact hash.

- [ ] **Step 5: Commit verified release documentation**

```bash
git add README.md docs/scaffold_model_card.md outputs/scaffold_release_audit.json
git commit -m "docs: publish scaffold generator verification"
```
