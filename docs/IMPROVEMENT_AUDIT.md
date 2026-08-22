# RNA Scaffold Generator V2 Improvement Audit

This matrix maps the V2 design requirements to shipped behavior and direct
verification. “Verified” means the code path has automated evidence; it
does not mean the CPU tiny model has useful biological performance.

| V2 requirement | Shipped behavior | Direct evidence |
|---|---|---|
| Canonical data contract | Strict A/U/C/G normalization, unique IDs/sequences, complete MMseqs2 assignments, expected 1,185 eligible clusters and 1,513/133/169 split counts | `tests/test_records_and_splits.py`, `tests/test_training_config.py` |
| Cluster-disjoint splitting | Complete connected components remain in one partition; leakage and incomplete assignments fail | `tests/test_records_and_splits.py` |
| Long-RNA retention | Source sequences are retained and dynamically cropped to a reproducible 512-nt epoch window | `test_long_rna_is_retained_and_cropped_reproducibly_per_epoch` |
| Bilateral flank targets | Targets are exactly motif start and target length minus motif end | `test_denoising_dataset_derives_bilateral_targets_without_v1_target_keys` |
| Geometry-free placement input | Placement receives only compact immutable motif tokens | `test_compact_fixed_motifs_removes_flank_length_and_position_leakage` |
| Joint legal flank selection | Pair scoring enforces both flank minima and requested/model/total limits | `tests/test_denoising_model.py`, `tests/test_iterative_decoding.py` |
| Prediction-only base loss | Fixed motif and padding positions are excluded from base loss and accuracy | `tests/test_model.py`, `tests/test_denoising_model.py` |
| RNA-FM none/frozen/LoRA modes | Explicit mode validation; no legacy boolean switch | `tests/test_pretrained_adapter.py`, `tests/test_training_config.py` |
| RNA-FM LoRA boundary | Final-layer q/k/v/out projections receive input-dropout LoRA; base parameters stay frozen | `tests/test_pretrained_adapter.py` |
| LoRA execution correctness | RNA-FM functional attention switches to its module path and all adapters receive finite gradients | `test_rna_fm_functional_attention_switches_to_adapter_aware_module_path` |
| Pretrained provenance | Pinned RNA-FM package plus reviewed expected/actual weight hashes, adapter settings, and versions enter metadata | `tests/test_pretrained_adapter.py`, `tests/test_final_review_contracts.py` |
| Strict V2 checkpoint contract | Missing, older, boolean, float, and string versions are rejected by inference, CLI/config resume before `Trainer.fit`, and the Lightning restore hook; exact V2 round-trip is strict | `tests/test_checkpoint_loading.py`, `tests/test_training_resume.py` |
| Confidence remasking | All scaffold positions are re-predicted; lowest raw-confidence positions are re-masked monotonically | `tests/test_iterative_decoding.py` |
| Motif immutability | Motif is restored at every step and remains exact in final candidates | `tests/test_iterative_decoding.py`, `tests/test_generate.py` |
| Fresh final likelihood | Final resolved tokens are re-forwarded before normalized scoring | `tests/test_iterative_decoding.py` |
| Diversity controls | Duplicate/edit/k-mer/homopolymer/GC policies and bounded attempts report each rejection reason | `tests/test_generate.py` |
| No quota fabrication | Diversity shortfall remains a shortfall; existing sequences are never copied | `test_generation_shortfall_is_audited_after_bounded_duplicate_attempts` |
| Meaningful baselines | Uniform, first-order Markov, and motif-boundary-aware second-order Markov are distinct methods | `tests/test_scaffold_evaluation.py`, `tests/test_benchmark_cli.py` |
| Explicit checkpoint terminal states | Missing/failed checkpoint methods retain unavailable/failed state without heuristic fallback | `tests/test_benchmark_cli.py` |
| Paired uncertainty | Motif-paired bootstrap is reproducible and uses only common motif IDs | `tests/test_scaffold_evaluation.py` |
| RNAfold truthfulness | Optional adapter distinguishes not-run, unavailable, timeout, failure, parse error, and success | `tests/test_rnafold_validator.py`, `tests/test_benchmark_cli.py` |
| Atomic six-artifact benchmark | Sibling staging is fully written, fsynced, hash-validated, then directory-renamed; partial runs are never published | `tests/test_benchmark_cli.py`, `tests/test_final_review_contracts.py` |
| Runtime release boundary | Runtime Python/YAML are scanned for removed placement/loss interfaces, old fallback wording, and removed training entry points | `tests/test_package_boundary.py`, `tests/test_release_audit.py` |
| V2 configuration boundary | Three matched training producers and the formal benchmark consumer are statically cross-validated; unsupported legacy configs are absent | `tests/test_training_config.py`, `tests/test_release_audit.py` |
| CPU end-to-end release gate | One optimizer step writes a V2 checkpoint, public resume and load gates validate it, V1 resume paths are rejected, and at least two unique motif-preserved candidates are sequence-evaluated without RNAfold | `test_cpu_tiny_v2_workflow_trains_loads_generates_and_evaluates` |

## Reproducibility artifacts

The authoritative setup writes:

- `outputs/server_setup/canonical_data_audit.json`: canonical input
  SHA-256 values, split assignments, record/cluster counts, thresholds, and
  data audit;
- `outputs/server_setup/rna_fm_sha256.txt`: the exact cached RNA-FM weight
  hash;
- `outputs/server_setup/scaffold_release_audit.json`: release checks,
  environment, repository state, and configuration hashes.

Every formal benchmark manifest additionally records command arguments,
seeds, software versions, configuration/data/checkpoint hashes, method
states, and hashes of the six output artifacts.

## Remaining external evidence

Local and CPU-container verification establishes interface, provenance, and
runtime correctness. Final scientific completion still requires matched
full-data A800 runs for no RNA-FM, frozen RNA-FM, and LoRA configurations,
followed by the formal equal-budget benchmark and appropriate experimental
validation. No smoke result is a folding-quality or biological-function
claim.
