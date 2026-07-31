import math
import json
from types import SimpleNamespace

import torch
import pytest

from evaluate_3d import (
    check_quality_gates,
    c1_lddt,
    cif_reference_fingerprint,
    discover_label_model_indices,
    evaluate_multi_reference_dataset,
    evaluate_prediction,
    file_sha256,
    filter_dataset_by_split_manifest,
    json_safe,
    kabsch_rmsd,
    load_multi_reference_labels,
    parse_recycle_counts,
    prediction_physical_metrics,
    reference_geometry_metrics,
    recycle_stability_metrics,
    summarize_metric_rows,
    validate_split_manifest_sequence_fingerprint,
)
from rna_scaffold_3d.geometry import apply_random_rigid_augmentation
from rna_scaffold_3d.rna_atoms import RNA_ATOM_TO_INDEX, RNA_NUM_ATOMS
from rna_scaffold_3d.rhofold import RhoFoldConfig, RhoFoldModel
from rna_scaffold_3d.splitting import sequence_dataset_fingerprint


def test_independent_metrics_are_rigid_transform_invariant():
    torch.manual_seed(21)
    target = torch.randn(1, 6, RNA_NUM_ATOMS, 3)
    mask = torch.ones(target.shape[:-1], dtype=torch.bool)
    transformed = apply_random_rigid_augmentation(target, mask)
    c1 = RNA_ATOM_TO_INDEX["C1'"]

    assert kabsch_rmsd(transformed[0, :, c1], target[0, :, c1]) < 1e-4
    assert c1_lddt(transformed[0, :, c1], target[0, :, c1]) == 100.0


def test_recycle_stability_metrics_detect_nonrigid_drift():
    c1 = RNA_ATOM_TO_INDEX["C1'"]
    final = torch.zeros(1, 5, RNA_NUM_ATOMS, 3)
    final[0, :, c1, 0] = torch.arange(5) * 5.9
    rigid = final.clone()
    rigid[..., 0] += 10.0
    drifted = final.clone()
    drifted[0, 2, c1, 1] += 3.0

    metrics = recycle_stability_metrics(
        {
            1: {"coords": rigid},
            2: {"coords": drifted},
            3: {"coords": final},
        }
    )

    assert metrics["recycle_1_to_3_c1_kabsch_rmsd"] < 1e-4
    assert metrics["recycle_2_to_3_c1_kabsch_rmsd"] > 0.1
    assert metrics["recycle_c1_kabsch_rmsd_max"] == (
        metrics["recycle_2_to_3_c1_kabsch_rmsd"]
    )
    assert metrics["recycle_c1_distance_rmsd_max"] > 0.0


def test_recycle_count_parser_rejects_silent_clamping():
    assert parse_recycle_counts("all", 3) == (1, 2, 3)
    assert parse_recycle_counts("3,1,3", 3) == (1, 3)
    with pytest.raises(ValueError, match="between 1"):
        parse_recycle_counts("1,4", 3)


def test_evaluate_prediction_supports_single_c1_label_dataset():
    c1 = RNA_ATOM_TO_INDEX["C1'"]
    pred = torch.zeros(4, RNA_NUM_ATOMS, 3)
    target = torch.stack([torch.tensor([float(index) * 5.9, 0.0, 0.0]) for index in range(4)])
    pred[:, c1] = target
    mask = torch.tensor([True, True, True, True])
    metrics = evaluate_prediction(pred, target, mask, torch.full((4,), 80.0))

    assert metrics["kabsch_rmsd"] < 1e-5
    assert metrics["distance_rmsd"] < 1e-5
    assert metrics["c1_lddt"] == 100.0
    assert metrics["adjacent_c1_mean"] == torch.tensor(5.9).item()
    assert metrics["mean_plddt"] == 80.0


def test_quality_gates_report_structure_and_physical_scale_failures():
    summary = {
        "records": 1.0,
        "c1_lddt": 42.0,
        "kabsch_rmsd": 18.0,
        "adjacent_c1_mean": 1.2,
    }

    failures = check_quality_gates(
        summary,
        min_records=10,
        min_lddt=50.0,
        max_kabsch_rmsd=15.0,
        adjacent_c1_min=4.5,
        adjacent_c1_max=7.0,
    )

    assert len(failures) == 4
    assert any("records" in failure for failure in failures)
    assert any("c1_lddt" in failure for failure in failures)
    assert any("kabsch_rmsd" in failure for failure in failures)
    assert any("adjacent_c1_mean" in failure for failure in failures)


def test_metric_coverage_gate_rejects_means_from_too_few_targets():
    rows = [
        {"c1_lddt": 80.0},
        {"c1_lddt": float("nan")},
        {"c1_lddt": float("nan")},
    ]
    summary = summarize_metric_rows(rows, ("c1_lddt",))

    failures = check_quality_gates(
        summary,
        min_metric_coverage=0.9,
        min_lddt=50.0,
    )

    assert summary["c1_lddt"] == 80.0
    assert summary["c1_lddt_records"] == 1.0
    assert summary["c1_lddt_coverage"] == 1.0 / 3.0
    assert len(failures) == 1
    assert "c1_lddt_coverage" in failures[0]


def test_target_pass_fraction_rejects_catastrophic_minority_hidden_by_mean():
    records = [
        {"c1_lddt": 100.0, "kabsch_rmsd": 0.0}
        for _ in range(9)
    ] + [{"c1_lddt": 0.0, "kabsch_rmsd": 100.0}]
    summary = summarize_metric_rows(
        records, ("c1_lddt", "kabsch_rmsd")
    )

    failures = check_quality_gates(
        summary,
        records=records,
        min_target_pass_fraction=0.95,
        min_lddt=50.0,
        max_kabsch_rmsd=15.0,
    )

    assert summary["c1_lddt"] == 90.0
    assert summary["kabsch_rmsd"] == 10.0
    assert len(failures) == 2
    assert all("target_pass_fraction=0.9000" in item for item in failures)


def test_target_pass_fraction_counts_missing_metric_as_failure():
    records = [{"c1_lddt": 80.0}, {"c1_lddt": float("nan")}]
    summary = summarize_metric_rows(records, ("c1_lddt",))

    failures = check_quality_gates(
        summary,
        records=records,
        min_target_pass_fraction=0.9,
        min_lddt=50.0,
    )

    assert len(failures) == 1
    assert "target_pass_fraction=0.5000" in failures[0]


def test_max_metrics_remain_worst_target_values_after_summary():
    rows = [
        {"recycle_c1_kabsch_rmsd_max": 1.0},
        {"recycle_c1_kabsch_rmsd_max": 8.0},
    ]

    summary = summarize_metric_rows(
        rows, ("recycle_c1_kabsch_rmsd_max",)
    )

    assert summary["recycle_c1_kabsch_rmsd_max"] == 8.0
    failures = check_quality_gates(
        summary, max_recycle_c1_rmsd=5.0
    )
    assert len(failures) == 1
    assert "8.0000" in failures[0]


def test_split_manifest_filters_exact_validation_targets(tmp_path):
    class Dataset:
        records = [
            SimpleNamespace(target_id="R1"),
            SimpleNamespace(target_id="R2"),
            SimpleNamespace(target_id="R3"),
        ]

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            return self.records[index].target_id

    manifest = tmp_path / "split.json"
    manifest.write_text(
        json.dumps(
            {
                "train_target_ids": ["R1", "R3"],
                "val_target_ids": ["R2"],
            }
        ),
        encoding="utf-8",
    )

    selected = filter_dataset_by_split_manifest(Dataset(), manifest)

    assert len(selected) == 1
    assert selected[0] == "R2"


def test_split_manifest_rejects_overlap_and_missing_targets(tmp_path):
    class Dataset:
        records = [SimpleNamespace(target_id="R1")]

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            return self.records[index].target_id

    manifest = tmp_path / "split.json"
    manifest.write_text(
        json.dumps(
            {
                "train_target_ids": ["R1"],
                "val_target_ids": ["R1"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="leaks"):
        filter_dataset_by_split_manifest(Dataset(), manifest)

    manifest.write_text(
        json.dumps(
            {
                "train_target_ids": ["R1"],
                "val_target_ids": ["missing"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="absent"):
        filter_dataset_by_split_manifest(Dataset(), manifest)


def test_split_manifest_rejects_stale_sequence_fingerprint(tmp_path):
    sequences = tmp_path / "sequences.csv"
    sequences.write_text(
        "target_id,sequence\nR1,AAAA\nR2,CCCC\nR3,GGGG\n",
        encoding="utf-8",
    )
    ordered = [
        SimpleNamespace(target_id="R1", sequence="AAAA"),
        SimpleNamespace(target_id="R2", sequence="CCCC"),
        SimpleNamespace(target_id="R3", sequence="GGGG"),
    ]
    payload = {
        "format_version": 3,
        "dataset_fingerprint": sequence_dataset_fingerprint(ordered),
        "kmer_size": 8,
        "jaccard_threshold": 0.8,
        "candidate_strategy": "exhaustive_length_bounded",
        "train_indices": [0, 2],
        "val_indices": [1],
        "train_target_ids": ["R1", "R3"],
        "val_target_ids": ["R2"],
        "audit": {
            "num_records": 3,
            "num_clusters": 3,
            "train_records": 2,
            "val_records": 1,
            "exact_sequence_overlap": 0,
            "cross_split_pairs_checked": 2,
            "cross_split_audit_exhaustive": True,
            "max_cross_split_jaccard": 0.0,
        },
    }
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    validated = validate_split_manifest_sequence_fingerprint(
        manifest, sequences
    )
    assert validated["dataset_fingerprint"] == payload[
        "dataset_fingerprint"
    ]

    sequences.write_text(
        "target_id,sequence\nR1,AAAA\nR2,UUUU\nR3,GGGG\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dataset_fingerprint"):
        validate_split_manifest_sequence_fingerprint(
            manifest, sequences
        )

    sequences.write_text(
        "target_id,sequence\nR1,AAAA\nR2,CCCC\nR3,GGGG\n",
        encoding="utf-8",
    )
    payload["val_indices"] = [2]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        validate_split_manifest_sequence_fingerprint(
            manifest, sequences
        )


def test_evaluator_rejects_fingerprinted_manifest_with_near_duplicate_leak(
    tmp_path,
):
    shared = "ACGU" * 25
    near_duplicate = shared[:50] + "A" + shared[51:]
    records = [
        SimpleNamespace(target_id="1ABC_A", sequence=shared),
        SimpleNamespace(target_id="2DEF_A", sequence=near_duplicate),
        SimpleNamespace(target_id="3GHI_A", sequence="A" * 100),
        SimpleNamespace(target_id="4JKL_A", sequence="G" * 100),
    ]
    sequences = tmp_path / "sequences.csv"
    sequences.write_text(
        "target_id,sequence\n"
        + "\n".join(
            f"{record.target_id},{record.sequence}"
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )
    payload = {
        "format_version": 3,
        "dataset_fingerprint": sequence_dataset_fingerprint(records),
        "seed": 7,
        "val_fraction": 0.5,
        "kmer_size": 8,
        "jaccard_threshold": 0.8,
        "candidate_strategy": "exhaustive_length_bounded",
        "train_indices": [0, 2],
        "val_indices": [1, 3],
        "train_target_ids": ["1ABC_A", "3GHI_A"],
        "val_target_ids": ["2DEF_A", "4JKL_A"],
        "audit": {
            "num_records": 4,
            "num_clusters": 4,
            "train_records": 2,
            "val_records": 2,
            "exact_sequence_overlap": 0,
            "cross_split_pairs_checked": 4,
            "cross_split_audit_exhaustive": True,
            "max_cross_split_jaccard": 0.0,
        },
    }
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="near-duplicate"):
        validate_split_manifest_sequence_fingerprint(
            manifest, sequences
        )


def test_evaluation_artifacts_use_strict_json_and_file_hashes(tmp_path):
    source = tmp_path / "labels.csv"
    source.write_bytes(b"target,value\nR1,1\n")
    payload = {
        "summary": {
            "finite": 1.5,
            "missing": float("nan"),
            "overflow": float("inf"),
        },
        "records": [{"metric": float("-inf")}],
    }

    safe = json_safe(payload)
    encoded = json.dumps(safe, allow_nan=False)
    decoded = json.loads(encoded)

    assert decoded["summary"]["finite"] == 1.5
    assert decoded["summary"]["missing"] is None
    assert decoded["summary"]["overflow"] is None
    assert decoded["records"][0]["metric"] is None
    assert file_sha256(source) == (
        "be904fd5f198ea109af79a85bb53bf12a376992b84126a7e"
        "23b0e392303a10e8"
    )


def test_cif_reference_fingerprint_hashes_only_unique_evaluated_files(
    tmp_path,
):
    cif_dir = tmp_path / "PDB_RNA"
    cif_dir.mkdir()
    (cif_dir / "1abc.cif").write_bytes(b"first")
    (cif_dir / "2abc.cif").write_bytes(b"second")
    (cif_dir / "unused.cif").write_bytes(b"unused")

    first = cif_reference_fingerprint(
        cif_dir, ["2ABC_B", "1ABC_A", "1ABC_C"]
    )
    reordered = cif_reference_fingerprint(
        cif_dir, ["1ABC_C", "2ABC_B", "1ABC_A"]
    )
    (cif_dir / "unused.cif").write_bytes(b"changed but unused")
    unused_changed = cif_reference_fingerprint(
        cif_dir, ["1ABC_A", "2ABC_B"]
    )
    (cif_dir / "1abc.cif").write_bytes(b"changed reference")
    reference_changed = cif_reference_fingerprint(
        cif_dir, ["1ABC_A", "2ABC_B"]
    )

    assert first == reordered == unused_changed
    assert first["cif_file_count"] == 2
    assert [item["name"] for item in first["cif_files"]] == [
        "1abc.cif",
        "2abc.cif",
    ]
    assert (
        reference_changed["cif_content_hash"]
        != first["cif_content_hash"]
    )


def test_prediction_physical_metrics_detect_bond_break_and_clash():
    model = RhoFoldModel(
        RhoFoldConfig(
            d_model=16,
            pair_dim=8,
            msa_dim=8,
            nhead=4,
            pair_heads=2,
            num_e2e_layers=1,
            num_structure_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            equivariant_layers=0,
        )
    ).eval()
    input_ids = torch.tensor([1, 2, 3, 4])
    with torch.inference_mode():
        coords = model(input_ids.unsqueeze(0))[0]
    baseline = prediction_physical_metrics(coords, input_ids)

    broken = coords.clone()
    broken[0, RNA_ATOM_TO_INDEX["O5'"], 0] += 2.0
    broken[1, RNA_ATOM_TO_INDEX["P"], 0] += 2.0
    broken_metrics = prediction_physical_metrics(broken, input_ids)
    collided = coords.clone()
    collided[0, RNA_ATOM_TO_INDEX["N6"]] = collided[
        0, RNA_ATOM_TO_INDEX["O2'"]
    ]
    collision_metrics = prediction_physical_metrics(collided, input_ids)

    assert all(math.isfinite(value) for value in baseline.values())
    assert (
        broken_metrics["covalent_bond_rmse"]
        > baseline["covalent_bond_rmse"]
    )
    assert broken_metrics["o3_p_bond_rmse"] > baseline["o3_p_bond_rmse"]
    assert (
        collision_metrics["clash_penetration_rms"]
        > baseline["clash_penetration_rms"]
    )


def test_reference_geometry_metrics_are_periodic_and_rigid_invariant():
    model = RhoFoldModel(
        RhoFoldConfig(
            d_model=16,
            pair_dim=8,
            msa_dim=8,
            nhead=4,
            pair_heads=2,
            num_e2e_layers=1,
            num_structure_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            equivariant_layers=0,
        )
    ).eval()
    input_ids = torch.tensor([1, 2, 3, 4, 1])
    with torch.inference_mode():
        coords = model(input_ids.unsqueeze(0))[0]
    mask = torch.ones(coords.shape[:-1], dtype=torch.bool)
    transformed = apply_random_rigid_augmentation(
        coords.unsqueeze(0), mask.unsqueeze(0)
    )[0]

    metrics = reference_geometry_metrics(
        transformed, coords, mask, input_ids
    )

    assert metrics["torsion_periodic_error"] < 1e-6
    assert metrics["torsion_mae_deg"] < 1e-3
    assert metrics["sugar_pucker_mae_deg"] < 1e-3
    assert metrics["base_orientation_mae_deg"] < 1e-3


def test_reference_geometry_metrics_detect_torsion_and_pucker_errors():
    torch.manual_seed(29)
    target = torch.randn(6, RNA_NUM_ATOMS, 3)
    mask = torch.ones(target.shape[:-1], dtype=torch.bool)
    input_ids = torch.tensor([1, 2, 3, 4, 1, 2])
    perturbed = target.clone()
    perturbed[2, RNA_ATOM_TO_INDEX["O3'"]] += torch.tensor(
        [1.2, -0.7, 0.4]
    )
    perturbed[3, RNA_ATOM_TO_INDEX["C2'"]] += torch.tensor(
        [0.8, 0.5, -1.1]
    )
    perturbed[0, RNA_ATOM_TO_INDEX["C4"]] += torch.tensor(
        [0.4, -0.9, 1.1]
    )

    matching = reference_geometry_metrics(
        target, target, mask, input_ids
    )
    changed = reference_geometry_metrics(
        perturbed, target, mask, input_ids
    )

    assert matching["torsion_mae_deg"] < 1e-3
    assert matching["sugar_pucker_mae_deg"] < 1e-3
    assert matching["base_orientation_mae_deg"] < 1e-3
    assert changed["torsion_mae_deg"] > matching["torsion_mae_deg"]
    assert (
        changed["sugar_pucker_mae_deg"]
        > matching["sugar_pucker_mae_deg"]
    )
    assert (
        changed["base_orientation_mae_deg"]
        > matching["base_orientation_mae_deg"]
    )


def test_quality_gates_reject_unphysical_all_atom_geometry():
    summary = {
        "records": 12.0,
        "covalent_bond_rmse": 0.3,
        "clash_penetration_rms": 0.2,
        "o3_p_bond_rmse": 0.4,
        "torsion_mae_deg": 45.0,
        "sugar_pucker_mae_deg": 35.0,
        "base_orientation_mae_deg": 40.0,
        "recycle_c1_kabsch_rmsd_max": 8.0,
    }

    failures = check_quality_gates(
        summary,
        max_covalent_bond_rmse=0.1,
        max_clash_penetration_rms=0.05,
        max_o3_p_bond_rmse=0.1,
        max_torsion_mae_deg=30.0,
        max_sugar_pucker_mae_deg=25.0,
        max_base_orientation_mae_deg=25.0,
        max_recycle_c1_rmsd=5.0,
    )

    assert len(failures) == 7
    assert any("recycle_c1_kabsch_rmsd_max" in failure for failure in failures)


def test_multi_reference_evaluation_selects_matching_experimental_conformation(tmp_path):
    sequences = tmp_path / "validation_sequences.csv"
    labels = tmp_path / "validation_labels.csv"
    sequences.write_text("target_id,sequence\nR1,AAAA\n", encoding="utf-8")
    rows = ["ID,resname,resid,x_1,y_1,z_1,x_2,y_2,z_2"]
    for residue in range(1, 5):
        rows.append(
            f"R1_{residue},A,{residue},{10.0 * residue},0,0,"
            f"{5.9 * (residue - 1)},0,0"
        )
    labels.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert discover_label_model_indices(labels) == [1, 2]
    items = load_multi_reference_labels(
        sequences,
        labels,
        model_indices=[1, 2],
        max_records=None,
        max_sequence_length=20,
    )

    class MatchingModel:
        def __call__(self, input_ids, return_aux):
            coords = torch.zeros(1, 4, RNA_NUM_ATOMS, 3)
            c1 = RNA_ATOM_TO_INDEX["C1'"]
            coords[0, :, c1, 0] = torch.arange(4) * 5.9
            return {"coords": coords, "plddt": torch.full((1, 4), 100.0)}

    result = evaluate_multi_reference_dataset(MatchingModel(), items, device="cpu")

    assert result["records"][0]["reference_model"] == 2.0
    assert result["summary"]["c1_lddt"] == 100.0
    assert result["summary"]["c1_lddt_coverage"] == 1.0
    assert result["summary"]["mean_reference_count"] == 2.0


def test_multi_reference_evaluator_runs_each_requested_recycle_count():
    sequence = "AAAAA"
    reference_coords = torch.stack(
        [torch.tensor([5.9 * index, 0.0, 0.0]) for index in range(5)]
    )
    items = [
        {
            "target_id": "R1",
            "sequence": sequence,
            "references": [
                {
                    "model_index": 1,
                    "coords": reference_coords,
                    "mask": torch.ones(5, dtype=torch.bool),
                }
            ],
        }
    ]

    class RecycleModel:
        recycle_iters = 2

        def __init__(self):
            self.calls = []

        def __call__(self, input_ids, return_aux, recycle_iters):
            self.calls.append(recycle_iters)
            coords = torch.zeros(1, 5, RNA_NUM_ATOMS, 3)
            c1 = RNA_ATOM_TO_INDEX["C1'"]
            coords[0, :, c1] = reference_coords
            if recycle_iters == 1:
                coords[0, 2, c1, 1] += 2.0
            return {
                "coords": coords,
                "plddt": torch.full((1, 5), 80.0),
            }

    model = RecycleModel()
    result = evaluate_multi_reference_dataset(
        model,
        items,
        device="cpu",
        recycle_counts=(1, 2),
    )

    assert model.calls == [1, 2]
    assert result["summary"]["recycle_c1_kabsch_rmsd_max"] > 0.0
    assert result["summary"]["recycle_c1_kabsch_rmsd_max_coverage"] == 1.0
