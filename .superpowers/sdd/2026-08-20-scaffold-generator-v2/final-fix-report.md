# Scaffold Generator V2 Final Fix Report

Date: 2026-08-22

Baseline: `4631605a87245b4131b3858f7f8b36164bcf6948`

Implementation commit: `e3934c413c2be0a79f22d95e8afe62d2cfe4e7f7`

## Outcome

All code-level Important findings I1-I9 and Minor findings M1-M3 from
`final-review-report.md` are closed with automated evidence. This is not a
scientific release: canonical-data recounting, three full A800 runs, real
checkpoint manifests, the reviewed RNA-FM weight digest, the formal held-out
benchmark, and biological validation remain external evidence gates. The code
and audit fail closed instead of manufacturing those artifacts or claims.

Controller rulings are implemented as follows:

- The published runtime contains only the V2 `motif_denoising` data path and
  checkpoint-backed generation. Legacy flank/masked datasets, prompt/result/
  greedy helpers, and V1 aliases were deleted. Markov code is confined to the
  benchmark namespace.
- Formal prior and nearest-reference sequences are reconstructed from only the
  canonical cluster-disjoint train partition under the training split contract.
- Benchmark output is completed in a sibling staging directory and published by
  one directory rename only after six-file validation. A non-empty destination
  is still rejected without deleting unknown files.
- Inference/preflight use `torch.load(..., weights_only=True)`. Lightning resume
  is explicitly documented as trusted-only, not an untrusted-checkpoint sandbox.
- Local code completion is separated from A800 training and formal scientific
  evidence; both remain `not_run` in the release audit.

## TDD evidence by finding

The initial review-contract RED run was:

```text
docker exec rna-v2-test python -m pytest tests/test_final_review_contracts.py -q
16 failed
```

Each failure corresponded to the root named below. The final focused run is:

```text
docker exec rna-v2-test python -m pytest tests/test_final_review_contracts.py -q
20 passed in 4.78s
```

### I1 — legal geometry and 8-nt boundary: closed

- RED: an 8-nt `2 + 4 + 2` canvas was excluded when the soft preferred total
  was 24.
- GREEN: `rna_scaffold/geometry.py` is the shared fact source for minimum total,
  five legal-pair predicates, record eligibility, and legal pair tensors.
  Dataset, decoding, generation, and benchmark all use total-canvas semantics;
  the exact 8-nt boundary is covered.
- External recount: the configured 1,513/133/169 partitions and 1,185 eligible
  clusters predate the corrected eligibility formula. Canonical inputs are not
  present locally. Setup/training therefore retain fail-closed expected-count
  checks, while README/model card mark these counts as requiring server
  recertification and config update if the recount differs.

### I2 — single 512-nt hard limit: closed

- RED: model, dataset, and generation settings accepted 513; training config
  accepted `data.max_target_length > model.max_length`.
- GREEN: one `MAX_SCAFFOLD_LENGTH = 512` is enforced by model, dataset/datamodule,
  generation, checkpoint hyperparameters, benchmark geometry, and training
  config cross-validation. Tests accept 512 and reject 513/mismatched limits.

### I3 — leakage-safe and geometry-matched benchmark: closed

- RED: no formal partition reconstruction existed; prior/reference used the
  full corpus and baselines could use one-nt flanks.
- GREEN: `_load_benchmark_partitions` rebuilds the exact eligible canonical
  cluster split, verifies source/split expectations, and exposes disjoint
  train/validation/test records. Only train records feed `RnaTrainingPrior` and
  nearest-training similarity. Uniform, Markov, and checkpoint methods share
  min flank/min total/preferred total/max length; every candidate row is checked
  against the same geometry contract.

### I4 — learned-method identity: closed

- RED: the expected-identity validator was absent, so a LoRA checkpoint could be
  labeled no-RNA-FM/frozen.
- GREEN: formal checkpoint methods require expected pretrained kind/mode,
  decoder mode, and denoise steps. Loaded checkpoints expose actual V2 and
  pretrained metadata. Training-manifest and loaded-checkpoint identities are
  both compared. Mismatch publishes a complete terminal `failed` run whose
  manifest/model summary retains the actual identity. Formal schema requires
  exactly one LoRA iterative primary.

### I5 — matched producers and stable consumption: closed

- RED: no no-RNA-FM/frozen configs or training-manifest producer existed, and
  benchmark paths did not match Lightning's variable best filename.
- GREEN: three audited configs share data/model/trainer budgets aside from the
  intended pretrained mode and use independent checkpoint directories/W&B
  names. `train.write_training_manifest` atomically binds the best checkpoint
  path/hash, config hash, identity, split audit, and software versions.
  Formal benchmark resolves and re-hashes that manifest/checkpoint; static
  dry-run cross-validates every producer/consumer path.

### I6 — remove broken V1 runtime surface: closed

- RED: legacy datasets/helpers/aliases were public and the DataModule defaulted
  to an incompatible V1 task.
- GREEN: only `motif_denoising` is constructible; legacy datasets, prompt/result
  helpers, greedy/V1 aliases, and generation Markov helpers are deleted. The
  benchmark prior lives only in `rna_scaffold/benchmarking.py` and
  `benchmark_scaffolds.py`. Package-boundary and release scans cover the removed
  symbols/task names and public API.

### I7 — restricted inference/preflight loading: closed

- RED: the audit observed `weights_only=False`.
- GREEN: all inference/preflight checkpoint reads use `weights_only=True`, with
  Lightning-shaped V2 round-trip tests. Documentation identifies trainer resume
  as a trusted-only pickle path and does not describe preflight as a sandbox.

### I8 — run-level atomic benchmark publication: closed

- RED: injected failure on the third artifact left a partial non-empty target.
- GREEN: all payloads and the terminal manifest are written/fsynced in a unique
  sibling staging directory, exact filenames and payload hashes are verified,
  and one directory rename publishes the run. Injected write failure leaves no
  published artifacts. Existing non-empty destinations remain fail-closed.

### I9 — cached vectorized flank sampling: closed

- RED: three candidates rebuilt the Cartesian pair tensor three times and the
  excluded-pair path iterated GPU pairs through `.tolist()`.
- GREEN: generation builds one `FlankPairSampler` per motif. It caches pair
  tensors/scores/encoded IDs and updates a device boolean used-ID mask without a
  per-pair host loop. The 512-nt/256-selection CPU gate covers 127,765 legal
  pairs, uniqueness, and a 3-second internal upper bound; the observed focused
  test passed.

### M1 — README scoped ruff gate: closed

- RED: the authoritative final review recorded 17 diagnostics for the documented
  command (a separate local no-ignore diagnostic was not treated as the gate).
- GREEN:

```text
docker exec rna-v2-test python -m ruff check --ignore EXE002 rna_scaffold scripts tests train.py generate_scaffold.py validate_scaffolds.py benchmark_scaffolds.py
All checks passed!
```

### M2 — reproducible RNA-FM input: closed, reviewed digest externally pending

- RED: dependency was unpinned, no expected weight digest was required, and
  benchmark versions omitted RNA-FM/Lightning.
- GREEN: both dependency surfaces pin official `rna-fm==0.2.2`. RNA-FM configs
  require a reviewed SHA-256 (literal or `RNA_FM_EXPECTED_SHA256`), verify a local
  weight before load, and fail closed if absent/mismatched. Server setup validates
  exactly 64 hex characters and compares the downloaded asset. Checkpoint,
  training, and benchmark metadata record expected/actual weight provenance plus
  RNA-FM and Lightning versions.
- No weight hash was invented. An operator must supply the independently reviewed
  official asset digest before RNA-FM setup/training can run.

### M3 — formal held-out motif panel and evidence grade: closed

- RED: formal config had two handwritten motifs and no held-out provenance or
  minimum common-motif grade.
- GREEN: formal motifs are deterministically derived only from test-split records,
  de-duplicated and round-robin stratified by source length, GC, and motif length.
  Every row/manifest carries source partition, target ID, sequence SHA-256,
  coordinates, and strata. Formal config preregisters 48 motifs; the two-motif
  list is smoke-only. Paired comparisons report `common_motif_n`; fewer than the
  configured 20 are `exploratory_not_release_grade`.

## Final verification

All commands used the existing `rna-v2-test` container; no container or package
installation was created.

```text
docker exec rna-v2-test python -m pytest -q
181 passed, 4 warnings in 36.05s

docker exec rna-v2-test python scripts/verify_scaffold_release.py --output /tmp/final-release-audit-v2.json
V2 runtime boundary: passed
V2 configuration schemas: passed
CPU tiny V2 end-to-end: passed
pytest: passed
all three CLI help checks: passed
RNAfold: unavailable
server_training: not_run
formal_benchmark: not_run

docker exec rna-v2-test python benchmark_scaffolds.py --config configs/benchmark_scaffolds.yaml --dry-run
status: validated_external_inputs_pending
scientific_release_status: not_run
pending: canonical CSV, cluster manifest, reviewed RNA-FM weight, and three training manifests

docker exec rna-v2-test python benchmark_scaffolds.py --config configs/benchmark_scaffolds.yaml --output-dir /tmp/rna-v2-smoke-final-20260822 --smoke-test
status: completed
artifact count: 6
scientific_release_status: not_release_grade
motif-panel grade: exploratory_not_release_grade

git diff --check
passed (only platform LF/CRLF notices)
```

## Remaining external gates and concerns

1. Re-run the canonical data audit with corrected geometry, review the new
   eligible/split/cluster counts, and update expected counts only from that
   artifact.
2. Independently review and export the official RNA-FM weight SHA-256; the code
   intentionally cannot proceed without it.
3. Run matched no-RNA-FM, frozen RNA-FM, and LoRA training on the A800. Preserve
   their generated `training_manifest.json` files and checkpoint hashes.
4. Run the real 48-motif formal benchmark from a fresh output directory and
   review identity, missingness, common-motif N, runtime/memory, RNAfold status,
   and paired uncertainty before any release claim.
5. The final-four-layer RNA-FM LoRA module path still trades away its fused fast
   path; only real A800 measurements can accept its performance cost.
6. RNAfold was unavailable locally. This is reported rather than treated as
   evidence of structural plausibility.
