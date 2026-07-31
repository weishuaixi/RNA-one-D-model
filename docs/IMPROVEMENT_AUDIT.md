# RNA-one-D-model improvement audit

This matrix maps every requested improvement to its implementation and direct
verification. “Verified” means the current code path has a focused automated
test; it does not claim that the one-epoch local smoke model has useful folding
accuracy.

| Requested improvement | Current implementation | Direct evidence |
|---|---|---|
| Random rotation augmentation before training | Batched random proper rotation plus translation, enabled by all 3D configs | `test_random_rigid_augmentation_preserves_pairwise_distances` |
| Kabsch alignment before coordinate loss | Aligned coordinate loss uses masked Kabsch; raw-frame MSE is disabled by default | `test_kabsch_and_fape_are_invariant_to_global_rigid_transform` |
| True FAPE/local-frame primary loss | All-atom residue-frame FAPE is mandatory and has positive config weight | `test_all_atom_fape_detects_non_representative_atom_error`, config validation |
| SE(3)/E(3)-equivariant structure path | IPA uses residue frames; internal-coordinate refiners use invariant distances; rebuilding transforms all atoms equivariantly | `test_invariant_point_attention_is_invariant_to_global_rigid_transform`, `test_full_all_atom_structure_is_equivariant_to_initial_global_rotation` |
| Residue rigid frame + backbone torsions + pucker + base orientation | Seven torsions, sugar pseudorotation, SO(3) base orientation and CCD templates construct all heavy atoms | internal-coordinate round-trip, sugar closure and glycosidic geometry tests |
| Triangle multiplicative incoming/outgoing | Separate directed incoming and outgoing updates with third-edge masks | four directed/masked triangle multiplication tests |
| Triangle attention starting/ending node | Separate starting/ending modules with third-edge pair bias | four triangle-attention value/gradient tests |
| Pair transition | Masked pair MLP transition after triangle stack | `test_pair_transition_masks_padding_and_updates_valid_pairs` |
| Sequence attention with pair bias | Pair tensor supplies per-head attention logits | `test_sequence_attention_uses_pair_bias_and_masks_padding_edges` |
| Outer-product sequence-to-pair update | True cross-channel outer product, chunked without changing gradients | two outer-product tests |
| Inject 2D pair information into sequence | Pair-biased attention in every trunk block and masked pair-to-sequence recycle update | pair-bias and recycling preservation tests |
| Attention pooling instead of mean | Learned masked attention pooling over full pair rows | `test_pair_attention_pooling_ignores_padding_and_zeroes_empty_rows` |
| Structure module accesses full pair tensor | Pair pooling, pair-conditioned attention and IPA consume the complete directed tensor | `test_structure_module_coordinates_receive_full_pair_tensor_gradients` |
| Separate distance/direction/contact channels | Directed pair representation; omega/theta/phi and contact heads; only distogram logits are symmetrized | `test_directed_pair_channels_and_symmetric_distance_head_are_separate`, orientation/contact gradient test |
| Pair initialization and all updates use pair mask | Initialization, OPM, triangles, transition, recycling, pooling and structure attention all apply masks | padding invariance plus module-specific mask tests |
| Padding must not affect shorter sequence | No unmasked `pair.mean`; empty rows return zero; structure output is length invariant | `test_padding_does_not_change_valid_residue_outputs_or_pair_features` |
| Save initial seq/pair and recycle initial + previous features | Each recycle starts from initial embeddings and adds normalized previous seq, full pair and C1′ distance features | recycling information and padding tests |
| Configurable whole-recycle stop gradient | Every nonfinal recycle is jointly detached when configured | two graph-mode stop-gradient tests |
| Explicit C1′/C4′ residue representatives | C1′ is the reported/scored residue coordinate; C4′/C1′/glycosidic N define frames | representative-coordinate and sequence-aware-frame tests |
| Random recycle count during training | Uniformly samples 1..maximum only in training; evaluation uses requested/all maximum counts | `test_training_randomly_samples_recycle_counts_but_evaluation_uses_maximum` |
| Test stability at different recycle counts | Evaluator runs every supported count and reports rigid-invariant drift with a worst-target gate | recycle parser/stability/evaluation tests |
| Symmetric distance but asymmetric orientation/relative channels | Directed pair stays asymmetric; only final distance logits are averaged with transpose | directed-pair symmetry separation test |
| Add orientation heads | Pair omega/theta/phi heads plus final sugar-to-base SO(3) orientation supervision and metric | pair orientation tests, base orientation loss/metric tests |
| Periodic α,β,γ,δ,ε,ζ,χ targets | Direct normalized sin/cos targets with `1-cos(Δθ)`; base-specific N9/N1 χ | periodic, base-specific χ and cross-residue ε/ζ tests |
| O3′(i)-P(i+1), cross-residue angles and torsions | Exact 1.60 Å construction/loss, both adjacent bond angles, and target-derived α/ε/ζ | phosphodiester bond/angle and cross-residue torsion tests |
| Base planarity and sugar closure | Explicit differentiable geometry losses and target-independent evaluator metrics | planarity, sugar closure and release-gate tests |
| Exclude covalently connected atoms from clash | Base-specific within-residue bond graph and adjacent O3′-P exclusion | covalent-exclusion and nonbonded-clash tests |
| Leakage-safe independent validation | Exact length-bounded weighted-Jaccard clustering, exhaustive train×val and train×holdout audits | split v3/holdout v2 tests and real 4,127-sequence audit |
| Reproducible training and release artifacts | Checkpoint v5 stores RNG/DataLoader states, semantic fingerprint and exact split/holdout manifest hashes | uninterrupted-vs-resume and tamper-rejection tests |
| Independent structural release metrics | C1′ accuracy, confidence calibration, physical geometry, torsion, pucker, base orientation, recycle stability, coverage and per-target pass fraction | evaluator and release-pipeline tests |

## Remaining external evidence

The implementation and local execution path are verified. Final model-quality
completion still requires a new full-data A800 training run producing checkpoint
format v5, followed by both official multi-reference C1′ validation and strict
held-out all-atom evaluator v6 gates. The one-epoch, two-record Windows smoke is
only runtime/provenance evidence and is not folding-quality evidence.
