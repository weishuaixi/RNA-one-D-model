import csv
import json
import math
import sys
from pathlib import Path

import torch
from Bio.PDB import MMCIFParser, PDBParser

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluate_3d import prediction_physical_metrics
from rna_scaffold_3d.rna_atoms import RNA_ATOM_NAMES, RNA_ATOM_TO_INDEX
from rna_scaffold_3d.sequence import encode_rna_sequence

BASE = ROOT / ".codex_tmp" / "baseline"
OUT = ROOT / "outputs"
OURS = BASE / "ours_all"
METHODS = {
    "Our model": None,
    "AlphaFold 3": "af3.pdb",
    "NuFold": "nu.pdb",
    "RoseTTAFoldNA": "rf2na.pdb",
    "RhoFold+": "rho.pdb",
    "trRosettaRNA": "trRNA.pdb",
}


def load_structure(path):
    with path.open(encoding="utf-8", errors="ignore") as handle:
        first = handle.readline()
    parser = MMCIFParser(QUIET=True) if first.startswith(("#", "data_")) else PDBParser(QUIET=True)
    return parser.get_structure(path.stem, str(path))


def parse_c1(path):
    points = []
    for residue in load_structure(path).get_residues():
        for name in ("C1'", "C1*"):
            if name in residue:
                points.append(torch.tensor(residue[name].coord.copy()))
                break
    if not points:
        raise ValueError(f"No C1' atoms: {path}")
    return torch.stack(points)


def parse_all(path, length):
    items = []
    for residue in load_structure(path).get_residues():
        atoms = {
            atom.name.replace("*", "'"): atom.coord.copy()
            for atom in residue
            if atom.name.replace("*", "'") in RNA_ATOM_TO_INDEX
        }
        if atoms:
            items.append(atoms)
    items = items[:length]
    if len(items) != length:
        return None
    coords = torch.zeros(length, len(RNA_ATOM_NAMES), 3)
    for index, atoms in enumerate(items):
        for atom, xyz in atoms.items():
            coords[index, RNA_ATOM_TO_INDEX[atom]] = torch.tensor(xyz)
    return coords


def kabsch(prediction, target):
    prediction = prediction.float() - prediction.float().mean(0)
    target = target.float() - target.float().mean(0)
    u, _, vh = torch.linalg.svd(prediction.T @ target)
    correction = torch.eye(3)
    correction[-1, -1] = torch.det(vh.T @ u.T)
    rotation = vh.T @ correction @ u.T
    return prediction @ rotation.T, target


def structural_metrics(prediction, target):
    count = min(len(prediction), len(target))
    prediction, target = prediction[:count], target[:count]
    aligned, centered_target = kabsch(prediction, target)
    rmsd = torch.sqrt(((aligned - centered_target) ** 2).sum(-1).mean()).item()
    pred_dist = torch.cdist(prediction.float(), prediction.float())
    target_dist = torch.cdist(target.float(), target.float())
    mask = (target_dist < 15) & (~torch.eye(count, dtype=torch.bool))
    error = (pred_dist - target_dist).abs()
    denominator = mask.sum().item()
    lddt = (
        sum(((error < threshold) & mask).sum().item() for threshold in (0.5, 1.0, 2.0, 4.0))
        / (4 * denominator)
        * 100
        if denominator
        else float("nan")
    )
    distance_rmsd = torch.sqrt(((pred_dist - target_dist)[mask] ** 2).mean()).item() if denominator else float("nan")
    return {
        "c1_lddt": lddt,
        "c1_kabsch_rmsd": rmsd,
        "c1_distance_rmsd": distance_rmsd,
        "aligned_residues": count,
    }


sequences = {}
with (BASE / "monomer_test.csv").open(encoding="utf-8-sig", newline="") as handle:
    for row in csv.DictReader(handle):
        target_id = f"{row['PDB ID'].lower()}_{row['Asym. Chain ID']}"
        sequences[target_id] = row["Sequence (unmod.)"].strip().upper()

targets = sorted(
    (path.stem for path in OURS.glob("*.pdb") if path.stem in sequences),
    key=lambda target_id: (len(sequences[target_id]), target_id),
)
rows = []
errors = []
for target_id in targets:
    sequence = sequences[target_id]
    reference_path = BASE / "monomers" / target_id / "rcsb.pdb"
    if not reference_path.exists():
        errors.append({"target_id": target_id, "method": "Reference", "error": "missing rcsb.pdb"})
        continue
    reference = parse_c1(reference_path)
    for method, filename in METHODS.items():
        path = OURS / f"{target_id}.pdb" if filename is None else BASE / "monomers" / target_id / filename
        if not path.exists():
            errors.append({"target_id": target_id, "method": method, "error": "prediction file missing"})
            continue
        try:
            prediction = parse_c1(path)
            result = structural_metrics(prediction, reference)
            all_coords = parse_all(path, len(sequence))
            if all_coords is not None:
                result.update(prediction_physical_metrics(all_coords, torch.tensor(encode_rna_sequence(sequence))))
            rows.append(
                {
                    "target_id": target_id,
                    "method": method,
                    "length": len(sequence),
                    "pred_c1_count": len(prediction),
                    "ref_c1_count": len(reference),
                    **result,
                }
            )
        except Exception as exc:
            errors.append({"target_id": target_id, "method": method, "error": repr(exc)})

numeric = [
    "c1_lddt",
    "c1_kabsch_rmsd",
    "c1_distance_rmsd",
    "covalent_bond_rmse",
    "backbone_angle_rmse_deg",
    "clash_penetration_rms",
    "base_planarity_rms",
    "sugar_closure_rmse",
    "o3_p_bond_rmse",
]
available_by_target = {}
for row in rows:
    available_by_target.setdefault(row["target_id"], set()).add(row["method"])
common_targets = {
    target_id for target_id, methods in available_by_target.items() if methods == set(METHODS)
}
summary = []
for method in METHODS:
    available_subset = [row for row in rows if row["method"] == method]
    subset = [row for row in available_subset if row["target_id"] in common_targets]
    if not subset:
        continue
    item = {"method": method, "n": len(subset), "available_n": len(available_subset)}
    for key in numeric:
        values = [row[key] for row in subset if key in row and math.isfinite(row[key])]
        if values:
            values.sort()
            middle = len(values) // 2
            median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
            item[f"{key}_mean"] = sum(values) / len(values)
            item[f"{key}_median"] = median
    summary.append(item)

OUT.mkdir(exist_ok=True)
per_target_path = OUT / "rnagym_baseline_per_target.csv"
summary_path = OUT / "rnagym_baseline_summary.csv"
errors_path = OUT / "rnagym_baseline_errors.csv"
with per_target_path.open("w", newline="", encoding="utf-8-sig") as handle:
    fields = sorted({key for row in rows for key in row})
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
    fields = ["method", "n", "available_n"] + sorted({key for row in summary for key in row} - {"method", "n", "available_n"})
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary)
with errors_path.open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=["target_id", "method", "error"])
    writer.writeheader()
    writer.writerows(errors)
(OUT / "rnagym_baseline_results.json").write_text(
    json.dumps({"targets": targets, "common_targets": sorted(common_targets), "summary": summary, "errors": errors}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps({"targets": len(targets), "rows": len(rows), "errors": len(errors), "summary": summary}, ensure_ascii=False, indent=2))
