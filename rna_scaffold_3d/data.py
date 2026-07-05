from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from rna_scaffold.utils import validate_rna_sequence


RNA_BASE_TO_ID = {"A": 1, "U": 2, "C": 3, "G": 4}
RNA3D_PAD_ID = 0
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
    ) -> "StanfordRna3DDataset":
        return cls(
            load_stanford_rna_3d_records(
                sequences_csv=sequences_csv,
                labels_csv=labels_csv,
                max_records=max_records,
                model_index=model_index,
            )
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        input_ids = torch.tensor([RNA_BASE_TO_ID[base] for base in record.sequence], dtype=torch.long)
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
) -> list[StanfordRna3DRecord]:
    sequences = _load_sequences(sequences_csv, max_records=max_records)
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
        if mask.any():
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


def _load_sequences(path: str | Path, max_records: int | None) -> dict[str, str]:
    sequences: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            target_id = (row.get("target_id") or "").strip()
            sequence = (row.get("sequence") or "").strip().upper().replace("T", "U")
            if not target_id or not validate_rna_sequence(sequence):
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
