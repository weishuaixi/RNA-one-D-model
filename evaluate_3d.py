from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from types import SimpleNamespace
from pathlib import Path

import torch
from torch.utils.data import Subset

from fold_3d import _load_model
from rna_scaffold_3d.data import StanfordRna3DDataset, StanfordRnaAllAtomDataset
from rna_scaffold_3d.losses import (
    base_relative_orientation_targets,
    base_planarity_loss,
    bond_angle_loss,
    bond_length_loss,
    rna_torsion_targets,
    steric_clash_loss,
    sugar_pucker_phase,
    sugar_ring_closure_loss,
)
from rna_scaffold_3d.rna_atoms import (
    RNA_ATOM_TO_INDEX,
    chemical_atom_mask,
)
from rna_scaffold_3d.rhofold import RHO_FOLD_ARCHITECTURE_VERSION
from rna_scaffold_3d.sequence import encode_rna_sequence
from rna_scaffold_3d.splitting import (
    SPLIT_FORMAT_VERSION,
    sequence_dataset_fingerprint,
    validate_split_manifest_partitions,
)

_PHYSICAL_METRIC_NAMES = (
    "covalent_bond_rmse",
    "backbone_angle_rmse_deg",
    "clash_penetration_rms",
    "base_planarity_rms",
    "sugar_closure_rmse",
    "o3_p_bond_rmse",
)
_REFERENCE_GEOMETRY_METRIC_NAMES = (
    "torsion_periodic_error",
    "torsion_mae_deg",
    "sugar_pucker_mae_deg",
    "base_orientation_mae_deg",
)
EVALUATION_FORMAT_VERSION = 6


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cif_reference_fingerprint(
    cif_dir: str | Path,
    target_ids: list[str],
) -> dict[str, object]:
    """Hash exactly the unique mmCIF files used by an evaluation report."""
    root = Path(cif_dir).resolve()
    pdb_ids = sorted(
        {
            target_id.rsplit("_", 1)[0].lower()
            for target_id in target_ids
        }
    )
    aggregate = hashlib.sha256()
    files: list[dict[str, str]] = []
    for pdb_id in pdb_ids:
        name = f"{pdb_id}.cif"
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Evaluated reference CIF disappeared before provenance "
                f"hashing: {path}"
            )
        content_hash = file_sha256(path)
        aggregate.update(name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(content_hash.encode("ascii"))
        aggregate.update(b"\n")
        files.append({"name": name, "sha256": content_hash})
    return {
        "cif_content_hash": aggregate.hexdigest(),
        "cif_content_hash_algorithm": (
            "sha256(sorted(filename + NUL + file_sha256 + LF))"
        ),
        "cif_file_count": len(files),
        "cif_files": files,
    }


def json_safe(value):
    """Convert non-finite floats to JSON null without changing gate inputs."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def _representative_target(
    target: torch.Tensor,
    coord_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target.ndim == 3:
        c1 = RNA_ATOM_TO_INDEX["C1'"]
        return target[:, c1], coord_mask[:, c1]
    if target.ndim == 2:
        return target, coord_mask
    raise ValueError("target must have shape [L, 3] or [L, A, 3].")


def kabsch_rmsd(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.size(0) == 0:
        return float("nan")
    pred_centered = pred - pred.mean(dim=0)
    target_centered = target - target.mean(dim=0)
    if pred.size(0) < 3:
        return float(torch.sqrt((pred_centered - target_centered).pow(2).sum(dim=-1).mean()))
    covariance = pred_centered.transpose(0, 1) @ target_centered
    u, _, vh = torch.linalg.svd(covariance.float(), full_matrices=False)
    correction = torch.eye(3, device=pred.device)
    correction[-1, -1] = torch.det(vh.transpose(-2, -1) @ u.transpose(-2, -1))
    rotation = vh.transpose(-2, -1) @ correction @ u.transpose(-2, -1)
    aligned = pred_centered.float() @ rotation.transpose(-2, -1)
    return float(torch.sqrt((aligned - target_centered.float()).pow(2).sum(dim=-1).mean()))


def distance_rmsd(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.size(0) < 2:
        return float("nan")
    difference = torch.cdist(pred.float(), pred.float()) - torch.cdist(target.float(), target.float())
    return float(torch.sqrt(difference.pow(2).mean()))


def parse_recycle_counts(value: str, maximum: int) -> tuple[int, ...]:
    """Parse explicit recycle counts without silently clamping requests."""
    if maximum < 1:
        raise ValueError("maximum recycle count must be positive.")
    if value.strip().lower() == "all":
        return tuple(range(1, maximum + 1))
    try:
        counts = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as error:
        raise ValueError("recycle counts must be comma-separated integers or 'all'.") from error
    if not counts or counts[0] < 1 or counts[-1] > maximum:
        raise ValueError(
            f"recycle counts must be between 1 and checkpoint maximum {maximum}."
        )
    return counts


def recycle_metric_names(recycle_counts: tuple[int, ...]) -> tuple[str, ...]:
    if len(recycle_counts) < 2:
        return ()
    final_count = recycle_counts[-1]
    names: list[str] = []
    for count in recycle_counts[:-1]:
        names.extend(
            (
                f"recycle_{count}_to_{final_count}_c1_kabsch_rmsd",
                f"recycle_{count}_to_{final_count}_c1_distance_rmsd",
            )
        )
    names.extend(
        (
            "recycle_c1_kabsch_rmsd_max",
            "recycle_c1_distance_rmsd_max",
        )
    )
    return tuple(names)


def recycle_stability_metrics(
    outputs: dict[int, dict[str, torch.Tensor]],
) -> dict[str, float]:
    """Measure rigid-invariant C1' drift from each recycle count to final."""
    counts = tuple(sorted(outputs))
    if len(counts) < 2:
        return {}
    c1 = RNA_ATOM_TO_INDEX["C1'"]
    final_count = counts[-1]
    final_points = outputs[final_count]["coords"][0, :, c1].float().cpu()
    kabsch_values: list[float] = []
    distance_values: list[float] = []
    metrics: dict[str, float] = {}
    for count in counts[:-1]:
        points = outputs[count]["coords"][0, :, c1].float().cpu()
        kabsch_value = kabsch_rmsd(points, final_points)
        distance_value = distance_rmsd(points, final_points)
        metrics[f"recycle_{count}_to_{final_count}_c1_kabsch_rmsd"] = kabsch_value
        metrics[f"recycle_{count}_to_{final_count}_c1_distance_rmsd"] = distance_value
        kabsch_values.append(kabsch_value)
        distance_values.append(distance_value)
    metrics["recycle_c1_kabsch_rmsd_max"] = max(kabsch_values)
    metrics["recycle_c1_distance_rmsd_max"] = max(distance_values)
    return metrics


def predict_recycles(
    model,
    input_ids: torch.Tensor,
    recycle_counts: tuple[int, ...] | None,
) -> dict[int, dict[str, torch.Tensor]]:
    if recycle_counts is None:
        maximum = int(getattr(model, "recycle_iters", 1))
        return {maximum: model(input_ids=input_ids, return_aux=True)}
    return {
        count: model(
            input_ids=input_ids,
            return_aux=True,
            recycle_iters=count,
        )
        for count in recycle_counts
    }


def c1_lddt(pred: torch.Tensor, target: torch.Tensor, cutoff: float = 15.0) -> float:
    if pred.size(0) < 2:
        return float("nan")
    pred_distance = torch.cdist(pred.float(), pred.float())
    target_distance = torch.cdist(target.float(), target.float())
    identity = torch.eye(pred.size(0), dtype=torch.bool, device=pred.device)
    neighborhood = ~identity & target_distance.lt(cutoff)
    if not neighborhood.any():
        return float("nan")
    error = (pred_distance - target_distance).abs()
    score = torch.stack([(error < threshold).float() for threshold in (0.5, 1.0, 2.0, 4.0)]).mean(0)
    return float(score[neighborhood].mean() * 100.0)


def per_residue_c1_lddt(
    pred: torch.Tensor,
    target: torch.Tensor,
    cutoff: float = 15.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local lDDT values and a mask for residues with valid neighbors."""
    pred_distance = torch.cdist(pred.float(), pred.float())
    target_distance = torch.cdist(target.float(), target.float())
    identity = torch.eye(pred.size(0), dtype=torch.bool, device=pred.device)
    neighborhood = ~identity & target_distance.lt(cutoff)
    error = (pred_distance - target_distance).abs()
    pair_score = torch.stack(
        [(error < threshold).float() for threshold in (0.5, 1.0, 2.0, 4.0)]
    ).mean(0)
    counts = neighborhood.sum(dim=-1)
    scores = (pair_score * neighborhood).sum(dim=-1) / counts.clamp(min=1)
    return scores * 100.0, counts.gt(0)


def prediction_physical_metrics(
    pred_coords: torch.Tensor,
    input_ids: torch.Tensor,
) -> dict[str, float]:
    """Return target-independent RNA all-atom geometry diagnostics."""
    if input_ids.ndim != 1 or pred_coords.ndim != 3:
        raise ValueError("Expected input_ids [L] and pred_coords [L, A, 3].")
    batched_coords = pred_coords.unsqueeze(0)
    batched_ids = input_ids.to(dtype=torch.long).unsqueeze(0)
    physical_mask = chemical_atom_mask(batched_ids)
    bond_rmse = torch.sqrt(
        bond_length_loss(batched_coords, physical_mask, batched_ids)
        .clamp_min(0.0)
    )
    angle_rmse = 10.0 * torch.sqrt(
        bond_angle_loss(
            batched_coords, physical_mask, batched_ids
        ).clamp_min(0.0)
    )
    clash_rms = torch.sqrt(
        steric_clash_loss(
            batched_coords, physical_mask, batched_ids
        ).clamp_min(0.0)
    )
    planarity_rms = torch.sqrt(
        base_planarity_loss(batched_coords, physical_mask).clamp_min(0.0)
    )
    sugar_rmse = torch.sqrt(
        sugar_ring_closure_loss(
            batched_coords, physical_mask
        ).clamp_min(0.0)
    )
    o3 = RNA_ATOM_TO_INDEX["O3'"]
    p = RNA_ATOM_TO_INDEX["P"]
    if pred_coords.size(0) >= 2:
        o3_p_distance = torch.linalg.norm(
            pred_coords[:-1, o3] - pred_coords[1:, p], dim=-1
        )
        o3_p_rmse = torch.sqrt(
            (o3_p_distance - 1.60).square().mean()
        )
    else:
        o3_p_rmse = pred_coords.sum() * 0.0
    return {
        "covalent_bond_rmse": float(bond_rmse),
        "backbone_angle_rmse_deg": float(angle_rmse),
        "clash_penetration_rms": float(clash_rms),
        "base_planarity_rms": float(planarity_rms),
        "sugar_closure_rmse": float(sugar_rmse),
        "o3_p_bond_rmse": float(o3_p_rmse),
    }


def base_orientation_error(
    pred_coords: torch.Tensor,
    target_coords: torch.Tensor,
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global-rigid-invariant sugar-to-base SO(3) errors per residue."""
    predicted_relative, predicted_valid = base_relative_orientation_targets(
        pred_coords.unsqueeze(0),
        pred_mask.unsqueeze(0),
        input_ids.unsqueeze(0),
    )
    target_relative, target_valid = base_relative_orientation_targets(
        target_coords.unsqueeze(0),
        target_mask.unsqueeze(0),
        input_ids.unsqueeze(0),
    )
    difference = (
        predicted_relative.transpose(-2, -1) @ target_relative
    )
    cosine = (
        (difference.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0
    ).clamp(-1.0, 1.0)
    skew = torch.stack(
        (
            difference[..., 2, 1] - difference[..., 1, 2],
            difference[..., 0, 2] - difference[..., 2, 0],
            difference[..., 1, 0] - difference[..., 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * torch.linalg.norm(skew, dim=-1)
    return torch.atan2(sine, cosine)[0], (predicted_valid & target_valid)[0]


def reference_geometry_metrics(
    pred_coords: torch.Tensor,
    target_coords: torch.Tensor,
    target_mask: torch.Tensor,
    input_ids: torch.Tensor,
) -> dict[str, float]:
    """Compare periodic RNA torsions and sugar pucker to an all-atom reference."""
    if (
        pred_coords.ndim != 3
        or target_coords.ndim != 3
        or target_mask.ndim != 2
        or input_ids.ndim != 1
    ):
        raise ValueError(
            "Expected pred/target [L,A,3], target_mask [L,A], and input_ids [L]."
        )
    if (
        pred_coords.shape != target_coords.shape
        or target_mask.shape != target_coords.shape[:-1]
        or input_ids.shape[0] != target_coords.shape[0]
    ):
        raise ValueError("Reference geometry inputs must have matching dimensions.")
    predicted = pred_coords.unsqueeze(0)
    target = target_coords.unsqueeze(0)
    ids = input_ids.to(dtype=torch.long).unsqueeze(0)
    predicted_mask = chemical_atom_mask(ids)
    observed_mask = target_mask.unsqueeze(0).bool()

    predicted_torsions, predicted_valid = rna_torsion_targets(
        predicted, predicted_mask, ids
    )
    target_torsions, target_valid = rna_torsion_targets(
        target, observed_mask, ids
    )
    torsion_valid = predicted_valid & target_valid
    torsion_cosine = (
        predicted_torsions * target_torsions
    ).sum(dim=-1).clamp(-1.0, 1.0)
    torsion_sine = (
        predicted_torsions[..., 0] * target_torsions[..., 1]
        - predicted_torsions[..., 1] * target_torsions[..., 0]
    ).abs()
    if torsion_valid.any():
        torsion_periodic = float(
            (1.0 - torsion_cosine)[torsion_valid].mean()
        )
        torsion_mae = float(
            torch.rad2deg(
                torch.atan2(
                    torsion_sine[torsion_valid],
                    torsion_cosine[torsion_valid],
                )
            ).mean()
        )
    else:
        torsion_periodic = float("nan")
        torsion_mae = float("nan")

    predicted_pucker, predicted_pucker_valid = sugar_pucker_phase(
        predicted, predicted_mask
    )
    target_pucker, target_pucker_valid = sugar_pucker_phase(
        target, observed_mask
    )
    pucker_valid = predicted_pucker_valid & target_pucker_valid
    pucker_cosine = (
        predicted_pucker * target_pucker
    ).sum(dim=-1).clamp(-1.0, 1.0)
    pucker_sine = (
        predicted_pucker[..., 0] * target_pucker[..., 1]
        - predicted_pucker[..., 1] * target_pucker[..., 0]
    ).abs()
    pucker_mae = (
        float(
            torch.rad2deg(
                torch.atan2(
                    pucker_sine[pucker_valid],
                    pucker_cosine[pucker_valid],
                )
            ).mean()
        )
        if pucker_valid.any()
        else float("nan")
    )
    orientation_error, orientation_valid = base_orientation_error(
        pred_coords,
        target_coords,
        predicted_mask[0],
        observed_mask[0],
        input_ids.to(dtype=torch.long),
    )
    orientation_mae = (
        float(torch.rad2deg(orientation_error[orientation_valid]).mean())
        if orientation_valid.any()
        else float("nan")
    )
    return {
        "torsion_periodic_error": torsion_periodic,
        "torsion_mae_deg": torsion_mae,
        "sugar_pucker_mae_deg": pucker_mae,
        "base_orientation_mae_deg": orientation_mae,
    }


def evaluate_prediction(
    pred_coords: torch.Tensor,
    target: torch.Tensor,
    coord_mask: torch.Tensor,
    predicted_plddt: torch.Tensor,
    input_ids: torch.Tensor | None = None,
) -> dict[str, float]:
    c1 = RNA_ATOM_TO_INDEX["C1'"]
    pred_points = pred_coords[:, c1]
    target_points, point_mask = _representative_target(target, coord_mask)
    pred_valid = pred_points[point_mask]
    target_valid = target_points[point_mask]
    adjacent_mask = point_mask[:-1] & point_mask[1:]
    adjacent = torch.linalg.norm(pred_points[1:] - pred_points[:-1], dim=-1)
    true_local_lddt, calibrated_mask = per_residue_c1_lddt(pred_valid, target_valid)
    predicted_confidence = predicted_plddt[point_mask][calibrated_mask].float()
    true_confidence = true_local_lddt[calibrated_mask]
    if predicted_confidence.numel():
        confidence_mae = float((predicted_confidence - true_confidence).abs().mean())
    else:
        confidence_mae = float("nan")
    if predicted_confidence.numel() >= 2:
        pred_centered = predicted_confidence - predicted_confidence.mean()
        true_centered = true_confidence - true_confidence.mean()
        denominator = torch.linalg.norm(pred_centered) * torch.linalg.norm(true_centered)
        confidence_correlation = (
            float((pred_centered * true_centered).sum() / denominator)
            if denominator > 1e-8
            else float("nan")
        )
    else:
        confidence_correlation = float("nan")
    metrics = {
        "valid_residues": float(point_mask.sum()),
        "kabsch_rmsd": kabsch_rmsd(pred_valid, target_valid),
        "distance_rmsd": distance_rmsd(pred_valid, target_valid),
        "c1_lddt": c1_lddt(pred_valid, target_valid),
        "adjacent_c1_mean": (
            float(adjacent[adjacent_mask].mean()) if adjacent_mask.any() else float("nan")
        ),
        "mean_plddt": float(predicted_plddt[point_mask].mean()) if point_mask.any() else float("nan"),
        "plddt_mae": confidence_mae,
        "plddt_correlation": confidence_correlation,
    }
    if input_ids is not None:
        metrics.update(prediction_physical_metrics(pred_coords, input_ids))
        if target.ndim == 3 and coord_mask.ndim == 2:
            metrics.update(
                reference_geometry_metrics(
                    pred_coords,
                    target,
                    coord_mask,
                    input_ids,
                )
            )
    return metrics


def summarize_metric_rows(
    rows: list[dict[str, float | str]],
    metric_names: tuple[str, ...],
) -> dict[str, float]:
    """Summarize finite values and expose per-metric target coverage."""
    total_records = len(rows)
    summary: dict[str, float] = {"records": float(total_records)}
    for name in metric_names:
        values = [
            float(row[name])
            for row in rows
            if math.isfinite(float(row[name]))
        ]
        valid_records = len(values)
        if not valid_records:
            summary[name] = float("nan")
        elif name.endswith("_max"):
            # A release gate named "max" must remain a worst-target gate
            # after dataset aggregation, not silently become a target mean.
            summary[name] = max(values)
        else:
            summary[name] = sum(values) / valid_records
        summary[f"{name}_records"] = float(valid_records)
        summary[f"{name}_coverage"] = (
            valid_records / total_records if total_records else 0.0
        )
    return summary


def filter_dataset_by_split_manifest(
    dataset,
    manifest_path: str | Path,
    partition: str = "val",
):
    """Select an auditable train/validation partition by persisted target ID."""
    if partition not in {"train", "val"}:
        raise ValueError("partition must be 'train' or 'val'.")
    records = getattr(dataset, "records", None)
    if records is None:
        raise ValueError("Split-manifest filtering requires dataset.records.")
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    train_ids = [str(value) for value in payload.get("train_target_ids", ())]
    val_ids = [str(value) for value in payload.get("val_target_ids", ())]
    if not train_ids or not val_ids:
        raise ValueError(
            "Split manifest must contain non-empty train_target_ids and "
            "val_target_ids."
        )
    if len(set(train_ids)) != len(train_ids) or len(set(val_ids)) != len(val_ids):
        raise ValueError("Split manifest contains duplicate target IDs.")
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise ValueError(
            "Split manifest leaks target IDs across train and validation: "
            + ", ".join(sorted(overlap)[:5])
        )
    record_indices: dict[str, int] = {}
    for index, record in enumerate(records):
        target_id = str(record.target_id)
        if target_id in record_indices:
            raise ValueError(
                f"Dataset contains duplicate target_id {target_id!r}."
            )
        record_indices[target_id] = index
    selected_ids = train_ids if partition == "train" else val_ids
    missing = [
        target_id for target_id in selected_ids
        if target_id not in record_indices
    ]
    if missing:
        raise ValueError(
            f"{partition} split contains {len(missing)} target(s) absent "
            "from the loaded dataset: "
            + ", ".join(missing[:5])
        )
    return Subset(
        dataset, [record_indices[target_id] for target_id in selected_ids]
    )


def validate_split_manifest_sequence_fingerprint(
    manifest_path: str | Path,
    sequences_csv: str | Path,
) -> dict[str, object]:
    """Prove a persisted split still describes the current target sequences."""
    payload = json.loads(
        Path(manifest_path).read_text(encoding="utf-8")
    )
    if payload.get("format_version") != SPLIT_FORMAT_VERSION:
        raise ValueError(
            "Split manifest format does not match the current splitter."
        )
    partitions = []
    for partition in ("train", "val"):
        indices = payload.get(f"{partition}_indices")
        target_ids = payload.get(f"{partition}_target_ids")
        if not isinstance(indices, list) or not isinstance(target_ids, list):
            raise ValueError(
                f"Split manifest must contain {partition}_indices and "
                f"{partition}_target_ids lists."
            )
        if len(indices) != len(target_ids):
            raise ValueError(
                f"Split manifest {partition} indices/target IDs differ in length."
            )
        partitions.extend(
            (int(index), str(target_id))
            for index, target_id in zip(indices, target_ids)
        )
    indices = [index for index, _ in partitions]
    if (
        len(set(indices)) != len(indices)
        or sorted(indices) != list(range(len(indices)))
    ):
        raise ValueError(
            "Split manifest indices must be unique and cover 0..N-1."
        )
    sequence_by_target: dict[str, str] = {}
    with Path(sequences_csv).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        if not {"target_id", "sequence"}.issubset(
            reader.fieldnames or ()
        ):
            raise ValueError(
                "Sequence CSV must contain target_id and sequence columns."
            )
        for row in reader:
            target_id = str(row.get("target_id") or "").strip()
            sequence = (
                str(row.get("sequence") or "")
                .strip()
                .upper()
                .replace("T", "U")
            )
            if target_id:
                if target_id in sequence_by_target:
                    raise ValueError(
                        f"Sequence CSV contains duplicate target_id "
                        f"{target_id!r}."
                    )
                sequence_by_target[target_id] = sequence
    ordered_records = []
    for _, target_id in sorted(partitions):
        if target_id not in sequence_by_target:
            raise ValueError(
                f"Split manifest target {target_id!r} is absent from "
                "the current sequence CSV."
            )
        ordered_records.append(
            SimpleNamespace(
                target_id=target_id,
                sequence=sequence_by_target[target_id],
            )
        )
    actual = sequence_dataset_fingerprint(ordered_records)
    expected = str(payload.get("dataset_fingerprint") or "")
    if not expected or actual != expected:
        raise ValueError(
            "Split manifest dataset_fingerprint does not match the "
            "current target sequences."
        )
    required_parameters = (
        "kmer_size",
        "jaccard_threshold",
        "candidate_strategy",
    )
    missing_parameters = [
        name for name in required_parameters if name not in payload
    ]
    if missing_parameters:
        raise ValueError(
            "Split manifest lacks leakage parameters: "
            + ", ".join(missing_parameters)
        )
    if payload["candidate_strategy"] != "exhaustive_length_bounded":
        raise ValueError(
            "Split manifest does not use exhaustive length-bounded candidates."
        )
    validate_split_manifest_partitions(
        payload,
        ordered_records,
        kmer_size=int(payload["kmer_size"]),
        jaccard_threshold=float(payload["jaccard_threshold"]),
    )
    return payload


@torch.inference_mode()
def evaluate_dataset(
    model,
    dataset,
    *,
    device: str | torch.device,
    max_records: int | None = None,
    recycle_counts: tuple[int, ...] | None = None,
) -> dict[str, object]:
    rows: list[dict[str, float | str]] = []
    count = len(dataset) if max_records is None else min(len(dataset), max_records)
    for index in range(count):
        item = dataset[index]
        sequence = str(item["sequence"])
        input_ids = torch.tensor([encode_rna_sequence(sequence)], dtype=torch.long, device=device)
        outputs = predict_recycles(model, input_ids, recycle_counts)
        output = outputs[max(outputs)]
        metrics = evaluate_prediction(
            output["coords"][0].float().cpu(),
            item["coords"].float().cpu(),
            item["coord_mask"].cpu(),
            output["plddt"][0].float().cpu(),
            input_ids[0].cpu(),
        )
        rows.append({
            "target_id": str(item["target_id"]),
            "length": float(len(sequence)),
            **metrics,
            **recycle_stability_metrics(outputs),
        })
    metric_names = (
        "kabsch_rmsd",
        "distance_rmsd",
        "c1_lddt",
        "adjacent_c1_mean",
        "mean_plddt",
        "plddt_mae",
        "plddt_correlation",
    ) + _PHYSICAL_METRIC_NAMES + _REFERENCE_GEOMETRY_METRIC_NAMES + recycle_metric_names(
        tuple(sorted(outputs)) if rows else (recycle_counts or ())
    )
    summary = summarize_metric_rows(rows, metric_names)
    return {"summary": summary, "records": rows}


def discover_label_model_indices(labels_csv: str | Path) -> list[int]:
    with Path(labels_csv).open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    indices = {
        int(match.group(1))
        for column in header
        if (match := re.fullmatch(r"x_(\d+)", column))
    }
    return sorted(indices)


def load_multi_reference_labels(
    sequences_csv: str | Path,
    labels_csv: str | Path,
    *,
    model_indices: list[int],
    max_records: int | None,
    max_sequence_length: int,
    min_reference_coverage: float = 0.5,
) -> list[dict[str, object]]:
    sequences: dict[str, str] = {}
    with Path(sequences_csv).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            target_id = str(row.get("target_id") or "").strip()
            sequence = str(row.get("sequence") or "").strip().upper().replace("T", "U")
            if target_id and sequence and len(sequence) <= max_sequence_length:
                sequences[target_id] = sequence
                if max_records is not None and len(sequences) >= max_records:
                    break
    references = {
        target_id: {
            index: {
                "coords": torch.zeros(len(sequence), 3),
                "mask": torch.zeros(len(sequence), dtype=torch.bool),
            }
            for index in model_indices
        }
        for target_id, sequence in sequences.items()
    }
    with Path(labels_csv).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label_id = str(row.get("ID") or "")
            target_id = label_id.rsplit("_", 1)[0]
            if target_id not in references:
                continue
            try:
                residue = int(float(row.get("resid") or "0")) - 1
            except ValueError:
                continue
            if residue < 0 or residue >= len(sequences[target_id]):
                continue
            for index in model_indices:
                try:
                    xyz = torch.tensor([float(row[f"{axis}_{index}"]) for axis in "xyz"])
                except (KeyError, TypeError, ValueError):
                    continue
                if torch.isfinite(xyz).all() and xyz.abs().max() < 1e17:
                    references[target_id][index]["coords"][residue] = xyz
                    references[target_id][index]["mask"][residue] = True
    items: list[dict[str, object]] = []
    for target_id, sequence in sequences.items():
        valid_references = [
            {"model_index": index, **reference}
            for index, reference in references[target_id].items()
            if float(reference["mask"].float().mean()) >= min_reference_coverage
        ]
        if valid_references:
            items.append(
                {
                    "target_id": target_id,
                    "sequence": sequence,
                    "references": valid_references,
                }
            )
    return items


@torch.inference_mode()
def evaluate_multi_reference_dataset(
    model,
    items: list[dict[str, object]],
    *,
    device: str | torch.device,
    recycle_counts: tuple[int, ...] | None = None,
) -> dict[str, object]:
    rows: list[dict[str, float | str]] = []
    for item in items:
        sequence = str(item["sequence"])
        input_ids = torch.tensor([encode_rna_sequence(sequence)], dtype=torch.long, device=device)
        outputs = predict_recycles(model, input_ids, recycle_counts)
        output = outputs[max(outputs)]
        candidates: list[dict[str, float | str]] = []
        for reference in item["references"]:
            metrics = evaluate_prediction(
                output["coords"][0].float().cpu(),
                reference["coords"],
                reference["mask"],
                output["plddt"][0].float().cpu(),
                input_ids[0].cpu(),
            )
            candidates.append(
                {
                    "reference_model": float(reference["model_index"]),
                    **metrics,
                }
            )
        best = max(
            candidates,
            key=lambda row: (
                float(row["c1_lddt"]) if math.isfinite(float(row["c1_lddt"])) else -1.0,
                -float(row["kabsch_rmsd"]) if math.isfinite(float(row["kabsch_rmsd"])) else -math.inf,
            ),
        )
        rows.append(
            {
                "target_id": str(item["target_id"]),
                "length": float(len(sequence)),
                "reference_count": float(len(candidates)),
                **best,
                **recycle_stability_metrics(outputs),
            }
        )
    metric_names = (
        "kabsch_rmsd",
        "distance_rmsd",
        "c1_lddt",
        "adjacent_c1_mean",
        "mean_plddt",
        "plddt_mae",
        "plddt_correlation",
    ) + _PHYSICAL_METRIC_NAMES + recycle_metric_names(
        tuple(sorted(outputs)) if rows else (recycle_counts or ())
    )
    summary = summarize_metric_rows(rows, metric_names)
    summary["mean_reference_count"] = (
        sum(float(row["reference_count"]) for row in rows) / len(rows) if rows else 0.0
    )
    return {"summary": summary, "records": rows}


def check_quality_gates(
    summary: dict[str, float],
    *,
    records: list[dict[str, object]] | None = None,
    min_records: int | None = None,
    min_metric_coverage: float | None = None,
    min_target_pass_fraction: float | None = None,
    min_lddt: float | None = None,
    max_kabsch_rmsd: float | None = None,
    adjacent_c1_min: float | None = None,
    adjacent_c1_max: float | None = None,
    max_plddt_mae: float | None = None,
    min_plddt_correlation: float | None = None,
    max_covalent_bond_rmse: float | None = None,
    max_backbone_angle_rmse_deg: float | None = None,
    max_clash_penetration_rms: float | None = None,
    max_base_planarity_rms: float | None = None,
    max_sugar_closure_rmse: float | None = None,
    max_o3_p_bond_rmse: float | None = None,
    max_torsion_mae_deg: float | None = None,
    max_sugar_pucker_mae_deg: float | None = None,
    max_base_orientation_mae_deg: float | None = None,
    max_recycle_c1_rmsd: float | None = None,
) -> list[str]:
    """Return reader-facing failures for explicitly requested release gates."""
    failures: list[str] = []
    if min_records is not None:
        actual_records = float(summary.get("records", float("nan")))
        if (
            not math.isfinite(actual_records)
            or actual_records < min_records
        ):
            failures.append(
                f"records={actual_records:.0f} must be >= {min_records}"
            )
    checks = (
        ("c1_lddt", min_lddt, lambda actual, threshold: actual >= threshold, ">="),
        ("kabsch_rmsd", max_kabsch_rmsd, lambda actual, threshold: actual <= threshold, "<="),
        ("adjacent_c1_mean", adjacent_c1_min, lambda actual, threshold: actual >= threshold, ">="),
        ("adjacent_c1_mean", adjacent_c1_max, lambda actual, threshold: actual <= threshold, "<="),
        ("plddt_mae", max_plddt_mae, lambda actual, threshold: actual <= threshold, "<="),
        ("plddt_correlation", min_plddt_correlation, lambda actual, threshold: actual >= threshold, ">="),
        ("covalent_bond_rmse", max_covalent_bond_rmse, lambda actual, threshold: actual <= threshold, "<="),
        ("backbone_angle_rmse_deg", max_backbone_angle_rmse_deg, lambda actual, threshold: actual <= threshold, "<="),
        ("clash_penetration_rms", max_clash_penetration_rms, lambda actual, threshold: actual <= threshold, "<="),
        ("base_planarity_rms", max_base_planarity_rms, lambda actual, threshold: actual <= threshold, "<="),
        ("sugar_closure_rmse", max_sugar_closure_rmse, lambda actual, threshold: actual <= threshold, "<="),
        ("o3_p_bond_rmse", max_o3_p_bond_rmse, lambda actual, threshold: actual <= threshold, "<="),
        ("torsion_mae_deg", max_torsion_mae_deg, lambda actual, threshold: actual <= threshold, "<="),
        ("sugar_pucker_mae_deg", max_sugar_pucker_mae_deg, lambda actual, threshold: actual <= threshold, "<="),
        ("base_orientation_mae_deg", max_base_orientation_mae_deg, lambda actual, threshold: actual <= threshold, "<="),
        (
            "recycle_c1_kabsch_rmsd_max",
            max_recycle_c1_rmsd,
            lambda actual, threshold: actual <= threshold,
            "<=",
        ),
    )
    checked_coverage: set[str] = set()
    for name, threshold, predicate, operator in checks:
        if threshold is None:
            continue
        if min_metric_coverage is not None and name not in checked_coverage:
            coverage = float(
                summary.get(f"{name}_coverage", float("nan"))
            )
            if (
                not math.isfinite(coverage)
                or coverage < min_metric_coverage
            ):
                failures.append(
                    f"{name}_coverage={coverage:.4f} must be >= "
                    f"{min_metric_coverage:.4f}"
                )
            checked_coverage.add(name)
        actual = float(summary.get(name, float("nan")))
        if not math.isfinite(actual) or not predicate(actual, threshold):
            failures.append(f"{name}={actual:.4f} must be {operator} {threshold:.4f}")
        if min_target_pass_fraction is not None:
            rows = records or []
            passing = 0
            for row in rows:
                try:
                    value = float(row.get(name, float("nan")))
                except (TypeError, ValueError):
                    value = float("nan")
                passing += int(
                    math.isfinite(value) and predicate(value, threshold)
                )
            pass_fraction = passing / len(rows) if rows else float("nan")
            if (
                not math.isfinite(pass_fraction)
                or pass_fraction < min_target_pass_fraction
            ):
                failures.append(
                    f"{name}_target_pass_fraction={pass_fraction:.4f} "
                    f"must be >= {min_target_pass_fraction:.4f} "
                    f"for target values {operator} {threshold:.4f}"
                )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a new RNA 3D checkpoint on held-out structures.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequences-csv", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--labels-csv", help="Stanford single-C1' validation labels.")
    source.add_argument("--cif-dir", help="Directory containing all-atom mmCIF files.")
    parser.add_argument(
        "--split-manifest",
        help=(
            "Training split manifest; with --cif-dir, evaluate only its "
            "validation target IDs."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-sequence-length", type=int, default=1536)
    parser.add_argument(
        "--label-model-indices",
        default="all",
        help="Comma-separated reference coordinate indices, or 'all' (default).",
    )
    parser.add_argument("--min-reference-coverage", type=float, default=0.5)
    parser.add_argument(
        "--recycle-counts",
        default="all",
        help="Comma-separated recycle counts or 'all' (default).",
    )
    parser.add_argument("--output", default="outputs/evaluation_3d.json")
    parser.add_argument("--require-min-records", type=int)
    parser.add_argument("--require-min-metric-coverage", type=float)
    parser.add_argument("--require-min-target-pass-fraction", type=float)
    parser.add_argument("--require-min-lddt", type=float)
    parser.add_argument("--require-max-kabsch-rmsd", type=float)
    parser.add_argument("--require-adjacent-c1-min", type=float)
    parser.add_argument("--require-adjacent-c1-max", type=float)
    parser.add_argument("--require-max-plddt-mae", type=float)
    parser.add_argument("--require-min-plddt-correlation", type=float)
    parser.add_argument("--require-max-covalent-bond-rmse", type=float)
    parser.add_argument("--require-max-backbone-angle-rmse-deg", type=float)
    parser.add_argument("--require-max-clash-penetration-rms", type=float)
    parser.add_argument("--require-max-base-planarity-rms", type=float)
    parser.add_argument("--require-max-sugar-closure-rmse", type=float)
    parser.add_argument("--require-max-o3-p-bond-rmse", type=float)
    parser.add_argument("--require-max-torsion-mae-deg", type=float)
    parser.add_argument("--require-max-sugar-pucker-mae-deg", type=float)
    parser.add_argument("--require-max-base-orientation-mae-deg", type=float)
    parser.add_argument("--require-max-recycle-c1-rmsd", type=float)
    args = parser.parse_args()
    if args.require_min_records is not None and args.require_min_records < 1:
        parser.error("--require-min-records must be at least 1.")
    if (
        args.require_min_metric_coverage is not None
        and not 0.0 <= args.require_min_metric_coverage <= 1.0
    ):
        parser.error("--require-min-metric-coverage must be between 0 and 1.")
    if (
        args.require_min_target_pass_fraction is not None
        and not 0.0 <= args.require_min_target_pass_fraction <= 1.0
    ):
        parser.error(
            "--require-min-target-pass-fraction must be between 0 and 1."
        )
    if args.split_manifest and not args.cif_dir:
        parser.error("--split-manifest currently requires --cif-dir.")

    model, _ = _load_model(args.checkpoint, args.device)
    try:
        recycle_counts = parse_recycle_counts(
            args.recycle_counts, int(model.recycle_iters)
        )
    except ValueError as error:
        parser.error(str(error))
    if args.labels_csv:
        model_indices = (
            discover_label_model_indices(args.labels_csv)
            if args.label_model_indices.lower() == "all"
            else [int(value) for value in args.label_model_indices.split(",")]
        )
        items = load_multi_reference_labels(
            args.sequences_csv,
            args.labels_csv,
            model_indices=model_indices,
            max_records=args.max_records,
            max_sequence_length=args.max_sequence_length,
            min_reference_coverage=args.min_reference_coverage,
        )
        result = evaluate_multi_reference_dataset(
            model,
            items,
            device=args.device,
            recycle_counts=recycle_counts,
        )
    else:
        split_target_ids: set[str] | None = None
        if args.split_manifest:
            split_payload = validate_split_manifest_sequence_fingerprint(
                args.split_manifest,
                args.sequences_csv,
            )
            split_target_ids = {
                str(value)
                for value in split_payload.get("val_target_ids", ())
            }
            if not split_target_ids:
                raise ValueError(
                    "Split manifest has no validation target IDs."
                )
        dataset = StanfordRnaAllAtomDataset.from_csv_and_cif(
            sequences_csv=args.sequences_csv,
            cif_dir=args.cif_dir,
            max_records=None if args.split_manifest else args.max_records,
            max_sequence_length=args.max_sequence_length,
            center_coordinates=True,
            target_ids=split_target_ids,
        )
        if args.split_manifest:
            dataset = filter_dataset_by_split_manifest(
                dataset, args.split_manifest, partition="val"
            )
        result = evaluate_dataset(
            model,
            dataset,
            device=args.device,
            max_records=args.max_records,
            recycle_counts=recycle_counts,
        )
    evaluation_parameters = {
        name: value
        for name, value in vars(args).items()
        if name not in {"checkpoint", "sequences_csv", "labels_csv", "cif_dir", "output"}
    }
    evaluation_parameters["resolved_recycle_counts"] = list(recycle_counts)
    data_metadata: dict[str, object] = {
        "sequences_csv": str(Path(args.sequences_csv).resolve()),
        "sequences_csv_sha256": file_sha256(args.sequences_csv),
    }
    if args.labels_csv:
        data_metadata.update({
            "labels_csv": str(Path(args.labels_csv).resolve()),
            "labels_csv_sha256": file_sha256(args.labels_csv),
        })
    else:
        target_ids = [
            str(record["target_id"])
            for record in result["records"]
        ]
        data_metadata.update({
            "cif_dir": str(Path(args.cif_dir).resolve()),
            **cif_reference_fingerprint(args.cif_dir, target_ids),
        })
        if args.split_manifest:
            data_metadata.update({
                "split_manifest": str(
                    Path(args.split_manifest).resolve()
                ),
                "split_manifest_sha256": file_sha256(
                    args.split_manifest
                ),
                "split_partition": "val",
            })
    result["metadata"] = {
        "evaluation_format_version": EVALUATION_FORMAT_VERSION,
        "architecture_version": RHO_FOLD_ARCHITECTURE_VERSION,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "data": data_metadata,
        "parameters": evaluation_parameters,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_result = json_safe(result)
    output_path.write_text(
        json.dumps(safe_result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(result["summary"]), indent=2, allow_nan=False))
    print(output_path)
    failures = check_quality_gates(
        result["summary"],
        records=result["records"],
        min_records=args.require_min_records,
        min_metric_coverage=args.require_min_metric_coverage,
        min_target_pass_fraction=args.require_min_target_pass_fraction,
        min_lddt=args.require_min_lddt,
        max_kabsch_rmsd=args.require_max_kabsch_rmsd,
        adjacent_c1_min=args.require_adjacent_c1_min,
        adjacent_c1_max=args.require_adjacent_c1_max,
        max_plddt_mae=args.require_max_plddt_mae,
        min_plddt_correlation=args.require_min_plddt_correlation,
        max_covalent_bond_rmse=args.require_max_covalent_bond_rmse,
        max_backbone_angle_rmse_deg=args.require_max_backbone_angle_rmse_deg,
        max_clash_penetration_rms=args.require_max_clash_penetration_rms,
        max_base_planarity_rms=args.require_max_base_planarity_rms,
        max_sugar_closure_rmse=args.require_max_sugar_closure_rmse,
        max_o3_p_bond_rmse=args.require_max_o3_p_bond_rmse,
        max_torsion_mae_deg=args.require_max_torsion_mae_deg,
        max_sugar_pucker_mae_deg=args.require_max_sugar_pucker_mae_deg,
        max_base_orientation_mae_deg=args.require_max_base_orientation_mae_deg,
        max_recycle_c1_rmsd=args.require_max_recycle_c1_rmsd,
    )
    if failures:
        raise SystemExit("Quality gates failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
