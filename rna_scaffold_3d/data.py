from __future__ import annotations

import csv
import math
import shlex
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from rna_scaffold_3d.rna_atoms import RNA_ATOM_TO_INDEX, RNA_NUM_ATOMS
from rna_scaffold_3d.sequence import RNA3D_PAD_ID, RNA_BASE_TO_ID, encode_rna_sequence, validate_rna_sequence


MISSING_COORD_ABS = 1e17


@dataclass(frozen=True)
class StanfordRna3DRecord:
    target_id: str
    sequence: str
    coords: torch.Tensor
    coord_mask: torch.Tensor


class StanfordRna3DDataset(Dataset):
    def __init__(self, records: list[StanfordRna3DRecord]) -> None:
        self.records = records

    @classmethod
    def from_csv(
        cls,
        sequences_csv: str | Path,
        labels_csv: str | Path,
        max_records: int | None = None,
        model_index: int = 1,
        max_sequence_length: int | None = None,
        min_coord_coverage: float = 0.0,
        center_coordinates: bool = False,
    ) -> "StanfordRna3DDataset":
        return cls(
            load_stanford_rna_3d_records(
                sequences_csv=sequences_csv,
                labels_csv=labels_csv,
                max_records=max_records,
                model_index=model_index,
                max_sequence_length=max_sequence_length,
                min_coord_coverage=min_coord_coverage,
                center_coordinates=center_coordinates,
            )
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        input_ids = torch.tensor(encode_rna_sequence(record.sequence), dtype=torch.long)
        return {
            "target_id": record.target_id,
            "sequence": record.sequence,
            "input_ids": input_ids,
            "coords": record.coords,
            "coord_mask": record.coord_mask,
        }


class StanfordRnaAllAtomDataset(Dataset):
    def __init__(self, records: list[StanfordRna3DRecord]) -> None:
        self.records = records

    @classmethod
    def from_csv_and_cif(
        cls,
        sequences_csv: str | Path,
        cif_dir: str | Path,
        max_records: int | None = None,
        max_sequence_length: int | None = None,
        min_atom_coverage: float = 0.5,
        center_coordinates: bool = True,
    ) -> "StanfordRnaAllAtomDataset":
        sequences = _load_sequences(
            sequences_csv,
            max_records=max_records,
            max_sequence_length=max_sequence_length,
        )
        records: list[StanfordRna3DRecord] = []
        cif_root = Path(cif_dir)
        for target_id, expected_sequence in sequences.items():
            pdb_id, chain_id = _split_target_id(target_id)
            cif_path = cif_root / f"{pdb_id.lower()}.cif"
            if not cif_path.exists():
                continue
            parsed = parse_cif_rna_chain(cif_path, chain_id=chain_id)
            if parsed is None:
                continue
            sequence, coords, atom_mask = parsed
            if sequence != expected_sequence:
                continue
            coverage = float(atom_mask.float().mean().item())
            if coverage < min_atom_coverage:
                continue
            if center_coordinates and atom_mask.any():
                coords = _center_valid_coordinates(coords, atom_mask)
            records.append(StanfordRna3DRecord(target_id, sequence, coords, atom_mask))
        return cls(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        input_ids = torch.tensor(encode_rna_sequence(record.sequence), dtype=torch.long)
        return {
            "target_id": record.target_id,
            "sequence": record.sequence,
            "input_ids": input_ids,
            "coords": record.coords,
            "coord_mask": record.coord_mask,
        }


def load_stanford_rna_3d_records(
    sequences_csv: str | Path,
    labels_csv: str | Path,
    max_records: int | None = None,
    model_index: int = 1,
    max_sequence_length: int | None = None,
    min_coord_coverage: float = 0.0,
    center_coordinates: bool = False,
) -> list[StanfordRna3DRecord]:
    sequences = _load_sequences(
        sequences_csv,
        max_records=max_records,
        max_sequence_length=max_sequence_length,
    )
    coord_rows = _load_label_coordinates(labels_csv, model_index=model_index, target_ids=set(sequences))
    records: list[StanfordRna3DRecord] = []
    for target_id, sequence in sequences.items():
        rows = coord_rows.get(target_id)
        if not rows:
            continue
        coords = torch.zeros((len(sequence), 3), dtype=torch.float32)
        mask = torch.zeros((len(sequence),), dtype=torch.bool)
        for resid, coord, valid in rows:
            index = resid - 1
            if 0 <= index < len(sequence):
                coords[index] = torch.tensor(coord, dtype=torch.float32)
                mask[index] = valid
        coverage = float(mask.float().mean().item()) if len(sequence) else 0.0
        if mask.any() and coverage >= min_coord_coverage:
            if center_coordinates:
                coords = _center_valid_coordinates(coords, mask)
            records.append(
                StanfordRna3DRecord(
                    target_id=target_id,
                    sequence=sequence,
                    coords=coords,
                    coord_mask=mask,
                )
            )
    return records


def collate_3d_batch(items: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    max_len = max(int(item["input_ids"].shape[0]) for item in items)  # type: ignore[index, union-attr]
    batch_size = len(items)
    input_ids = torch.full((batch_size, max_len), RNA3D_PAD_ID, dtype=torch.long)
    first_coords = items[0]["coords"]  # type: ignore[index]
    if first_coords.ndim == 3:
        coords = torch.zeros((batch_size, max_len, first_coords.shape[1], 3), dtype=torch.float32)
        coord_mask = torch.zeros((batch_size, max_len, first_coords.shape[1]), dtype=torch.bool)
    else:
        coords = torch.zeros((batch_size, max_len, 3), dtype=torch.float32)
        coord_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)
    target_ids: list[str] = []
    sequences: list[str] = []

    for row, item in enumerate(items):
        ids = item["input_ids"]  # type: ignore[assignment]
        item_coords = item["coords"]  # type: ignore[assignment]
        item_mask = item["coord_mask"]  # type: ignore[assignment]
        length = int(ids.shape[0])
        input_ids[row, :length] = ids
        coords[row, :length] = item_coords
        coord_mask[row, :length] = item_mask
        padding_mask[row, :length] = False
        target_ids.append(str(item["target_id"]))
        sequences.append(str(item["sequence"]))

    return {
        "target_ids": target_ids,
        "sequences": sequences,
        "input_ids": input_ids,
        "coords": coords,
        "coord_mask": coord_mask,
        "padding_mask": padding_mask,
    }


def _load_sequences(
    path: str | Path,
    max_records: int | None,
    max_sequence_length: int | None,
) -> dict[str, str]:
    sequences: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            target_id = (row.get("target_id") or "").strip()
            sequence = (row.get("sequence") or "").strip().upper().replace("T", "U")
            if not target_id or not validate_rna_sequence(sequence):
                continue
            if max_sequence_length is not None and len(sequence) > max_sequence_length:
                continue
            sequences[target_id] = sequence
            if max_records is not None and len(sequences) >= max_records:
                break
    return sequences


def _load_label_coordinates(
    path: str | Path,
    model_index: int,
    target_ids: set[str],
) -> dict[str, list[tuple[int, tuple[float, float, float], bool]]]:
    x_field = f"x_{model_index}"
    y_field = f"y_{model_index}"
    z_field = f"z_{model_index}"
    rows_by_target: dict[str, list[tuple[int, tuple[float, float, float], bool]]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label_id = (row.get("ID") or "").strip()
            target_id = label_id.rsplit("_", 1)[0]
            if target_id not in target_ids:
                continue
            resid_text = row.get("resid") or "0"
            try:
                resid = int(float(resid_text))
                coord = (
                    float(row[x_field] or "nan"),
                    float(row[y_field] or "nan"),
                    float(row[z_field] or "nan"),
                )
            except (KeyError, ValueError):
                continue
            valid = _is_valid_coord(coord)
            rows_by_target.setdefault(target_id, []).append((resid, coord if valid else (0.0, 0.0, 0.0), valid))
    for rows in rows_by_target.values():
        rows.sort(key=lambda item: item[0])
    return rows_by_target


def _is_valid_coord(coord: tuple[float, float, float]) -> bool:
    return all(math.isfinite(value) and abs(value) < MISSING_COORD_ABS for value in coord)


def _center_valid_coordinates(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    centered = coords.clone()
    center = centered[mask].mean(dim=0)
    centered[mask] = centered[mask] - center
    centered[~mask] = 0.0
    return centered


def _split_target_id(target_id: str) -> tuple[str, str]:
    pdb_id, chain_id = target_id.rsplit("_", 1)
    return pdb_id, chain_id


def parse_cif_rna_chain(path: str | Path, chain_id: str) -> tuple[str, torch.Tensor, torch.Tensor] | None:
    text = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    headers: list[str] = []
    rows: list[list[str]] = []
    in_atom_loop = False
    for raw_line in text:
        line = raw_line.strip()
        if not line:
            continue
        if line == "loop_":
            headers = []
            rows = []
            in_atom_loop = False
            continue
        if line.startswith("_"):
            if line.startswith("_atom_site."):
                headers.append(line)
                in_atom_loop = True
            elif in_atom_loop:
                break
            continue
        if in_atom_loop:
            if line.startswith("#"):
                break
            if headers:
                parts = shlex.split(line)
                if len(parts) >= len(headers):
                    rows.append(parts[: len(headers)])
    if not headers or not rows:
        return None

    field = {name.replace("_atom_site.", ""): index for index, name in enumerate(headers)}
    required = ["label_comp_id", "label_atom_id", "label_asym_id", "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z"]
    if any(name not in field for name in required):
        return None

    residues: dict[int, str] = {}
    atom_coords: dict[tuple[int, str], tuple[float, float, float]] = {}
    for row in rows:
        if row[field["label_asym_id"]] != chain_id:
            continue
        base = row[field["label_comp_id"]].upper().replace("T", "U")
        atom = row[field["label_atom_id"]]
        atom = atom.strip('"')
        if base not in RNA_BASE_TO_ID or atom not in RNA_ATOM_TO_INDEX:
            continue
        try:
            resid = int(float(row[field["label_seq_id"]]))
            coord = (
                float(row[field["Cartn_x"]]),
                float(row[field["Cartn_y"]]),
                float(row[field["Cartn_z"]]),
            )
        except ValueError:
            continue
        residues[resid] = base
        atom_coords[(resid, atom)] = coord
    if not residues:
        return None

    ordered_resids = sorted(residues)
    sequence = "".join(residues[resid] for resid in ordered_resids)
    resid_to_index = {resid: index for index, resid in enumerate(ordered_resids)}
    coords = torch.zeros((len(sequence), RNA_NUM_ATOMS, 3), dtype=torch.float32)
    mask = torch.zeros((len(sequence), RNA_NUM_ATOMS), dtype=torch.bool)
    for (resid, atom), coord in atom_coords.items():
        residue_index = resid_to_index[resid]
        atom_index = RNA_ATOM_TO_INDEX[atom]
        coords[residue_index, atom_index] = torch.tensor(coord, dtype=torch.float32)
        mask[residue_index, atom_index] = True
    return sequence, coords, mask
