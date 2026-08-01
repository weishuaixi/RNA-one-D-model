from __future__ import annotations

import csv
from difflib import SequenceMatcher
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from rna_scaffold_3d.rna_atoms import (
    RNA_ATOM_TO_INDEX,
    RNA_BACKBONE_ATOMS,
    RNA_NUM_ATOMS,
    chemical_atom_mask,
)
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
    def __init__(
        self,
        records: list[StanfordRna3DRecord],
        stats: dict[str, int | float | bool] | None = None,
    ) -> None:
        self.records = records
        self.stats = stats or {"accepted": len(records)}

    @classmethod
    def from_csv_and_cif(
        cls,
        sequences_csv: str | Path,
        cif_dir: str | Path,
        max_records: int | None = None,
        max_sequence_length: int | None = None,
        min_atom_coverage: float = 0.5,
        center_coordinates: bool = True,
        cache_path: str | Path | None = None,
        min_sequence_identity: float = 0.9,
        min_sequence_coverage: float = 0.9,
        target_ids: set[str] | None = None,
    ) -> "StanfordRnaAllAtomDataset":
        cache_metadata = None
        if cache_path is not None:
            cache_metadata = _all_atom_cache_metadata(
                sequences_csv=sequences_csv,
                cif_dir=cif_dir,
                max_records=max_records,
                max_sequence_length=max_sequence_length,
                min_atom_coverage=min_atom_coverage,
                center_coordinates=center_coordinates,
                min_sequence_identity=min_sequence_identity,
                min_sequence_coverage=min_sequence_coverage,
                target_ids=target_ids,
            )
            cached = _load_all_atom_cache(cache_path, cache_metadata)
            if cached is not None:
                return cls(cached, {"accepted": len(cached), "loaded_from_cache": True})
        sequences = _load_sequences(
            sequences_csv,
            max_records=None,
            max_sequence_length=max_sequence_length,
        )
        if target_ids is not None:
            allowed = {str(target_id) for target_id in target_ids}
            sequences = {
                target_id: sequence
                for target_id, sequence in sequences.items()
                if target_id in allowed
            }
        records: list[StanfordRna3DRecord] = []
        stats: dict[str, int | float | bool] = {
            "candidate_sequences": len(sequences),
            "missing_cif": 0,
            "parse_failed": 0,
            "alignment_failed": 0,
            "low_atom_coverage": 0,
            "exact_match": 0,
            "aligned_match": 0,
            "accepted": 0,
            "loaded_from_cache": False,
        }
        cif_root = Path(cif_dir)
        for target_id, expected_sequence in sequences.items():
            pdb_id, chain_id = _split_target_id(target_id)
            cif_path = cif_root / f"{pdb_id.lower()}.cif"
            if not cif_path.exists():
                stats["missing_cif"] = int(stats["missing_cif"]) + 1
                continue
            parsed = parse_cif_rna_chain(
                cif_path,
                chain_id=chain_id,
                expected_sequence=expected_sequence,
            )
            if parsed is None:
                stats["parse_failed"] = int(stats["parse_failed"]) + 1
                continue
            sequence, coords, atom_mask = parsed
            if sequence != expected_sequence:
                aligned = _align_parsed_structure(
                    expected_sequence,
                    sequence,
                    coords,
                    atom_mask,
                    min_identity=min_sequence_identity,
                    min_coverage=min_sequence_coverage,
                )
                if aligned is None:
                    stats["alignment_failed"] = int(stats["alignment_failed"]) + 1
                    continue
                sequence, coords, atom_mask = aligned
                stats["aligned_match"] = int(stats["aligned_match"]) + 1
            else:
                stats["exact_match"] = int(stats["exact_match"]) + 1
            coverage = _chemical_atom_coverage(sequence, atom_mask)
            if coverage < min_atom_coverage:
                stats["low_atom_coverage"] = int(stats["low_atom_coverage"]) + 1
                continue
            if center_coordinates and atom_mask.any():
                coords = _center_valid_coordinates(coords, atom_mask)
            records.append(StanfordRna3DRecord(target_id, sequence, coords, atom_mask))
            stats["accepted"] = len(records)
            if max_records is not None and len(records) >= max_records:
                break
        if cache_path is not None:
            assert cache_metadata is not None
            _save_all_atom_cache(cache_path, cache_metadata, records)
        return cls(records, stats)

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


def _align_parsed_structure(
    expected_sequence: str,
    parsed_sequence: str,
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
    min_identity: float = 0.9,
    min_coverage: float = 0.9,
) -> tuple[str, torch.Tensor, torch.Tensor] | None:
    """Globally align observed residues, using chain breaks to disambiguate gaps."""
    aligned_coords = torch.zeros(
        len(expected_sequence), coords.size(1), 3, dtype=coords.dtype
    )
    aligned_mask = torch.zeros(
        len(expected_sequence), atom_mask.size(1), dtype=torch.bool
    )
    matched_residues = 0
    try:
        import gemmi
    except ImportError:
        gemmi = None
    if gemmi is not None:
        scoring = gemmi.AlignmentScoring()
        target_gapo = [int(scoring.gapo)] * (len(parsed_sequence) + 1)
        o3 = RNA_ATOM_TO_INDEX["O3'"]
        p = RNA_ATOM_TO_INDEX["P"]
        c1 = RNA_ATOM_TO_INDEX["C1'"]
        for index in range(1, len(parsed_sequence)):
            phosphodiester_observed = (
                atom_mask[index - 1, o3] & atom_mask[index, p]
            )
            c1_observed = (
                atom_mask[index - 1, c1] & atom_mask[index, c1]
            )
            phosphodiester_break = (
                phosphodiester_observed
                and torch.linalg.norm(
                    coords[index - 1, o3] - coords[index, p]
                ).item() > 2.5
            )
            c1_break = (
                c1_observed
                and torch.linalg.norm(
                    coords[index - 1, c1] - coords[index, c1]
                ).item() > 10.0
            )
            if phosphodiester_break or c1_break:
                target_gapo[index] = 0
        result = gemmi.align_string_sequences(
            list(expected_sequence),
            list(parsed_sequence),
            target_gapo,
            scoring,
        )
        expected_aligned = result.add_gaps(expected_sequence, 1)
        parsed_aligned = result.add_gaps(parsed_sequence, 2)
        expected_index = 0
        parsed_index = 0
        backbone_count = len(RNA_BACKBONE_ATOMS)
        for expected_base, parsed_base in zip(
            expected_aligned, parsed_aligned
        ):
            if expected_base != "-" and parsed_base != "-":
                if expected_base == parsed_base:
                    aligned_coords[expected_index] = coords[parsed_index]
                    aligned_mask[expected_index] = atom_mask[parsed_index]
                    matched_residues += 1
                else:
                    aligned_coords[
                        expected_index, :backbone_count
                    ] = coords[parsed_index, :backbone_count]
                    aligned_mask[
                        expected_index, :backbone_count
                    ] = atom_mask[parsed_index, :backbone_count]
            if expected_base != "-":
                expected_index += 1
            if parsed_base != "-":
                parsed_index += 1
    else:  # pragma: no cover - Gemmi is a pinned runtime dependency.
        matcher = SequenceMatcher(
            None, expected_sequence, parsed_sequence, autojunk=False
        )
        for block in matcher.get_matching_blocks():
            if block.size == 0:
                continue
            expected_slice = slice(block.a, block.a + block.size)
            parsed_slice = slice(block.b, block.b + block.size)
            aligned_coords[expected_slice] = coords[parsed_slice]
            aligned_mask[expected_slice] = atom_mask[parsed_slice]
            matched_residues += block.size
    identity = matched_residues / max(1, max(len(expected_sequence), len(parsed_sequence)))
    coverage = matched_residues / max(1, len(expected_sequence))
    if identity < min_identity or coverage < min_coverage:
        return None
    return expected_sequence, aligned_coords, aligned_mask


def _chemical_atom_coverage(
    sequence: str,
    atom_mask: torch.Tensor,
) -> float:
    """Fraction of chemically expected atoms that were experimentally observed."""
    input_ids = torch.tensor(
        encode_rna_sequence(sequence),
        dtype=torch.long,
        device=atom_mask.device,
    )
    expected = chemical_atom_mask(input_ids)
    denominator = int(expected.sum().item())
    if denominator == 0:
        return 0.0
    observed = atom_mask.bool() & expected
    return float(observed.sum().item() / denominator)


_ALL_ATOM_CACHE_VERSION = 8


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relevant_cif_stat_digest(
    sequences_csv: str | Path,
    cif_root: Path,
    max_sequence_length: int | None,
    target_ids: set[str] | None,
) -> tuple[int, str]:
    sequences = _load_sequences(
        sequences_csv,
        max_records=None,
        max_sequence_length=max_sequence_length,
    )
    if target_ids is not None:
        allowed = {str(target_id) for target_id in target_ids}
        sequences = {
            target_id: sequence
            for target_id, sequence in sequences.items()
            if target_id in allowed
        }
    pdb_ids = sorted(
        {_split_target_id(target_id)[0].lower() for target_id in sequences}
    )
    digest = hashlib.sha256()
    present = 0
    for pdb_id in pdb_ids:
        path = cif_root / f"{pdb_id}.cif"
        try:
            stat = path.stat()
        except OSError:
            fingerprint = f"{pdb_id}.cif:missing\n"
        else:
            present += 1
            fingerprint = (
                f"{pdb_id}.cif:{stat.st_size}:{stat.st_mtime_ns}\n"
            )
        digest.update(fingerprint.encode("utf-8"))
    return present, digest.hexdigest()


def _all_atom_cache_metadata(
    *,
    sequences_csv: str | Path,
    cif_dir: str | Path,
    max_records: int | None,
    max_sequence_length: int | None,
    min_atom_coverage: float,
    center_coordinates: bool,
    min_sequence_identity: float,
    min_sequence_coverage: float,
    target_ids: set[str] | None = None,
) -> dict[str, object]:
    sequence_path = Path(sequences_csv).resolve()
    stat = sequence_path.stat()
    cif_root = Path(cif_dir).resolve()
    cif_count, cif_stat_digest = _relevant_cif_stat_digest(
        sequence_path,
        cif_root,
        max_sequence_length,
        target_ids,
    )
    metadata = {
        "version": _ALL_ATOM_CACHE_VERSION,
        "sequences_csv": str(sequence_path),
        "sequences_size": stat.st_size,
        "sequences_mtime_ns": stat.st_mtime_ns,
        "sequences_sha256": _file_sha256(sequence_path),
        "cif_dir": str(cif_root),
        "cif_count": cif_count,
        "relevant_cif_stat_sha256": cif_stat_digest,
        "max_records": max_records,
        "max_sequence_length": max_sequence_length,
        "min_atom_coverage": float(min_atom_coverage),
        "center_coordinates": bool(center_coordinates),
        "min_sequence_identity": float(min_sequence_identity),
        "min_sequence_coverage": float(min_sequence_coverage),
    }
    if target_ids is not None:
        metadata["target_ids"] = sorted(
            str(target_id) for target_id in target_ids
        )
    return metadata


def _load_all_atom_cache(
    cache_path: str | Path,
    expected_metadata: dict[str, object],
) -> list[StanfordRna3DRecord] | None:
    path = Path(cache_path)
    if not path.exists():
        return None
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch < 2.0
            payload = torch.load(path, map_location="cpu")
    except (OSError, RuntimeError, EOFError):
        return None
    if not isinstance(payload, dict) or payload.get("metadata") != expected_metadata:
        return None
    serialized = payload.get("records")
    if not isinstance(serialized, list):
        return None
    records: list[StanfordRna3DRecord] = []
    for item in serialized:
        if not isinstance(item, dict):
            return None
        try:
            records.append(
                StanfordRna3DRecord(
                    target_id=str(item["target_id"]),
                    sequence=str(item["sequence"]),
                    coords=item["coords"].float(),
                    coord_mask=item["coord_mask"].bool(),
                )
            )
        except (KeyError, AttributeError):
            return None
    return records


def _save_all_atom_cache(
    cache_path: str | Path,
    metadata: dict[str, object],
    records: list[StanfordRna3DRecord],
) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "metadata": metadata,
        "records": [
            {
                "target_id": record.target_id,
                "sequence": record.sequence,
                "coords": record.coords.cpu(),
                "coord_mask": record.coord_mask.cpu(),
            }
            for record in records
        ],
    }
    torch.save(payload, temporary)
    temporary.replace(path)


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
    example_weights = torch.ones(batch_size, dtype=torch.float32)
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
        example_weights[row] = float(item.get("example_weight", 1.0))
        target_ids.append(str(item["target_id"]))
        sequences.append(str(item["sequence"]))

    return {
        "target_ids": target_ids,
        "sequences": sequences,
        "input_ids": input_ids,
        "coords": coords,
        "coord_mask": coord_mask,
        "padding_mask": padding_mask,
        "example_weights": example_weights,
    }


def _load_sequences(
    path: str | Path,
    max_records: int | None,
    max_sequence_length: int | None,
) -> dict[str, str]:
    sequences: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline().strip().split(",")
        try:
            target_idx = header.index("target_id")
            sequence_idx = header.index("sequence")
        except ValueError as exc:
            raise ValueError(f"CSV must contain target_id and sequence columns: {path}") from exc
        required_prefix = max(target_idx, sequence_idx) + 1
        for line in handle:
            fields = line.rstrip("\n\r").split(",", required_prefix)
            if len(fields) < required_prefix:
                continue
            target_id = fields[target_idx].strip()
            sequence = fields[sequence_idx].strip().upper().replace("T", "U")
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


def _cif_loop_rows(
    lines: list[str],
    category: str,
) -> tuple[list[str], list[list[str]]]:
    """Read one simple mmCIF loop for fields whose values contain no spaces."""
    index = 0
    while index < len(lines):
        if lines[index].strip() != "loop_":
            index += 1
            continue
        index += 1
        headers: list[str] = []
        while index < len(lines) and lines[index].strip().startswith("_"):
            headers.append(lines[index].strip())
            index += 1
        if not any(header.startswith(category) for header in headers):
            continue
        rows: list[list[str]] = []
        while index < len(lines):
            line = lines[index].strip()
            if (
                not line
                or line.startswith("#")
                or line == "loop_"
                or line.startswith("_")
                or line.startswith("data_")
            ):
                break
            parts = line.split()
            if len(parts) >= len(headers):
                rows.append(parts[:len(headers)])
            index += 1
        return headers, rows
    return [], []


def _component_parent_mapping(
    rows: list[dict[str, str]],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in rows:
        component = _strip_cif_quotes(row.get("id", "")).upper()
        parent = _strip_cif_quotes(
            row.get("mon_nstd_parent_comp_id", "")
        ).upper().replace("T", "U")
        if component and parent in RNA_BASE_TO_ID:
            mapping[component] = parent
    return mapping


def _infer_base_from_atom_names(atom_names: set[str]) -> str | None:
    """Infer a canonical parent only from diagnostic nucleobase heavy atoms."""
    if "N9" in atom_names:
        if "N6" in atom_names and "O6" not in atom_names:
            return "A"
        if "O6" in atom_names or "N2" in atom_names:
            return "G"
    if "N1" in atom_names:
        if "N4" in atom_names and "O4" not in atom_names:
            return "C"
        if "O4" in atom_names:
            return "U"
    return None


def _assemble_cif_rna_atoms_for_rows(
    atom_rows: list[dict[str, str]],
    component_parents: dict[str, str] | None = None,
) -> tuple[str, torch.Tensor, torch.Tensor] | None:
    """Select one model and one coherent alternate conformation per residue."""
    component_parents = component_parents or {}
    if not atom_rows:
        return None

    def model_number(row: dict[str, str]) -> int:
        try:
            return int(float(row.get("pdbx_PDB_model_num", "1")))
        except ValueError:
            return 1

    selected_model = min(model_number(row) for row in atom_rows)
    residue_rows: dict[int, list[dict[str, object]]] = {}
    for row in atom_rows:
        if model_number(row) != selected_model:
            continue
        component = _strip_cif_quotes(
            row.get("label_comp_id", "")
        ).upper()
        base = component.replace("T", "U")
        if base not in RNA_BASE_TO_ID:
            base = component_parents.get(component, "")
        atom = _strip_cif_quotes(row.get("label_atom_id", ""))
        if atom not in RNA_ATOM_TO_INDEX:
            continue
        try:
            residue = int(float(row.get("label_seq_id", "")))
            coord = (
                float(row.get("Cartn_x", "nan")),
                float(row.get("Cartn_y", "nan")),
                float(row.get("Cartn_z", "nan")),
            )
            occupancy = float(row.get("occupancy", "1"))
        except ValueError:
            continue
        if not _is_valid_coord(coord) or not math.isfinite(occupancy) or occupancy <= 0:
            continue
        alternate = _strip_cif_quotes(
            row.get("label_alt_id", ".")
        )
        if alternate in {".", "?"}:
            alternate = ""
        residue_rows.setdefault(residue, []).append(
            {
                "base": base,
                "component": component,
                "atom": atom,
                "coord": coord,
                "occupancy": occupancy,
                "alternate": alternate,
            }
        )
    if not residue_rows:
        return None

    residues: dict[int, str] = {}
    atom_coords: dict[tuple[int, str], tuple[float, float, float]] = {}
    for residue, candidates in residue_rows.items():
        alternate_scores: dict[str, float] = {}
        for candidate in candidates:
            alternate = str(candidate["alternate"])
            if alternate:
                alternate_scores[alternate] = (
                    alternate_scores.get(alternate, 0.0)
                    + float(candidate["occupancy"])
                )
        chosen_alternate = (
            min(
                alternate_scores,
                key=lambda alternate: (
                    -alternate_scores[alternate],
                    alternate,
                ),
            )
            if alternate_scores
            else ""
        )
        eligible = [
            candidate for candidate in candidates
            if candidate["alternate"] in {"", chosen_alternate}
        ]
        inferred_base = _infer_base_from_atom_names(
            {str(candidate["atom"]) for candidate in eligible}
        )
        base_scores: dict[str, float] = {}
        for candidate in eligible:
            base = str(candidate["base"])
            if base not in RNA_BASE_TO_ID:
                continue
            base_scores[base] = (
                base_scores.get(base, 0.0)
                + float(candidate["occupancy"])
            )
        if not base_scores and inferred_base is not None:
            base_scores[inferred_base] = sum(
                float(candidate["occupancy"])
                for candidate in eligible
            )
        if not base_scores:
            continue
        selected_base = min(
            base_scores,
            key=lambda base: (-base_scores[base], base),
        )
        residues[residue] = selected_base
        selected_atoms: dict[str, dict[str, object]] = {}
        for candidate in eligible:
            candidate_base = str(candidate["base"])
            if candidate_base not in {"", selected_base}:
                continue
            atom = str(candidate["atom"])
            current = selected_atoms.get(atom)
            rank = (
                candidate["alternate"] == chosen_alternate
                and bool(chosen_alternate),
                float(candidate["occupancy"]),
            )
            current_rank = (
                current["alternate"] == chosen_alternate
                and bool(chosen_alternate),
                float(current["occupancy"]),
            ) if current is not None else (False, -1.0)
            if current is None or rank > current_rank:
                selected_atoms[atom] = candidate
        for atom, candidate in selected_atoms.items():
            atom_coords[(residue, atom)] = candidate["coord"]  # type: ignore[assignment]

    if not residues:
        return None
    ordered_residues = sorted(residues)
    sequence = "".join(residues[residue] for residue in ordered_residues)
    index_by_residue = {
        residue: index for index, residue in enumerate(ordered_residues)
    }
    coords = torch.zeros(
        (len(sequence), RNA_NUM_ATOMS, 3), dtype=torch.float32
    )
    mask = torch.zeros(
        (len(sequence), RNA_NUM_ATOMS), dtype=torch.bool
    )
    for (residue, atom), coord in atom_coords.items():
        residue_index = index_by_residue[residue]
        atom_index = RNA_ATOM_TO_INDEX[atom]
        coords[residue_index, atom_index] = torch.tensor(
            coord, dtype=torch.float32
        )
        mask[residue_index, atom_index] = True
    return sequence, coords, mask


def _sequence_candidate_score(
    expected_sequence: str,
    parsed: tuple[str, torch.Tensor, torch.Tensor],
) -> tuple[bool, float, float, float, int, int]:
    sequence, _, atom_mask = parsed
    try:
        import gemmi
    except ImportError:  # pragma: no cover - Gemmi is a pinned dependency.
        matched = sum(
            block.size
            for block in SequenceMatcher(
                None, expected_sequence, sequence, autojunk=False
            ).get_matching_blocks()
        )
    else:
        scoring = gemmi.AlignmentScoring()
        result = gemmi.align_string_sequences(
            list(expected_sequence),
            list(sequence),
            [1] * (len(sequence) + 1),
            scoring,
        )
        expected_aligned = result.add_gaps(expected_sequence, 1)
        parsed_aligned = result.add_gaps(sequence, 2)
        matched = sum(
            expected_base == parsed_base
            for expected_base, parsed_base in zip(
                expected_aligned, parsed_aligned
            )
            if expected_base != "-" and parsed_base != "-"
        )
    identity = matched / max(1, max(len(expected_sequence), len(sequence)))
    coverage = matched / max(1, len(expected_sequence))
    atom_coverage = _chemical_atom_coverage(sequence, atom_mask)
    return (
        sequence == expected_sequence,
        identity,
        coverage,
        atom_coverage,
        int(atom_mask.sum().item()),
        -abs(len(sequence) - len(expected_sequence)),
    )


def _assemble_cif_rna_atoms(
    atom_rows: list[dict[str, str]],
    chain_id: str,
    component_parents: dict[str, str] | None = None,
    expected_sequence: str | None = None,
) -> tuple[str, torch.Tensor, torch.Tensor] | None:
    """Resolve label/auth chain aliases, then assemble the best RNA chain."""
    label_matches = [
        row
        for row in atom_rows
        if _strip_cif_quotes(row.get("label_asym_id", "")) == chain_id
    ]
    auth_matches = [
        row
        for row in atom_rows
        if _strip_cif_quotes(row.get("auth_asym_id", "")) == chain_id
    ]
    candidate_rows = [label_matches]
    if auth_matches != label_matches:
        candidate_rows.append(auth_matches)
    candidates = [
        parsed
        for rows in candidate_rows
        if (
            parsed := _assemble_cif_rna_atoms_for_rows(
                rows, component_parents
            )
        )
        is not None
    ]
    if not candidates:
        return None
    if expected_sequence is None or len(candidates) == 1:
        return candidates[0]
    return max(
        candidates,
        key=lambda parsed: _sequence_candidate_score(
            expected_sequence, parsed
        ),
    )


def parse_cif_rna_chain(
    path: str | Path,
    chain_id: str,
    expected_sequence: str | None = None,
) -> tuple[str, torch.Tensor, torch.Tensor] | None:
    try:
        import gemmi
    except ImportError:
        gemmi = None
    if gemmi is not None:
        try:
            parsed = _parse_cif_with_gemmi(
                path,
                chain_id,
                gemmi,
                expected_sequence=expected_sequence,
            )
            if parsed is not None:
                return parsed
        except (OSError, RuntimeError, ValueError):
            # Keep the dependency optional and retain the tested text fallback.
            pass

    text = Path(path).read_text(
        encoding="utf-8", errors="ignore"
    ).splitlines()
    headers, rows = _cif_loop_rows(text, "_atom_site.")
    if not headers or not rows:
        return None
    field = {
        name.replace("_atom_site.", ""): index
        for index, name in enumerate(headers)
    }
    required = ["label_comp_id", "label_atom_id", "label_asym_id", "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z"]
    if any(name not in field for name in required):
        return None
    atom_rows = [
        {
            name: row[index]
            for name, index in field.items()
        }
        for row in rows
    ]
    chem_headers, chem_rows = _cif_loop_rows(text, "_chem_comp.")
    chem_field = {
        name.replace("_chem_comp.", ""): index
        for index, name in enumerate(chem_headers)
    }
    component_parents = _component_parent_mapping(
        [
            {
                name: row[index]
                for name, index in chem_field.items()
            }
            for row in chem_rows
        ]
    )
    return _assemble_cif_rna_atoms(
        atom_rows,
        chain_id,
        component_parents,
        expected_sequence=expected_sequence,
    )


def _parse_cif_with_gemmi(
    path: str | Path,
    chain_id: str,
    gemmi_module,
    expected_sequence: str | None = None,
):
    """Fast standards-compliant atom_site parsing when Gemmi is installed."""
    block = gemmi_module.cif.read_file(str(path)).sole_block()
    tags = [
        "_atom_site.label_comp_id",
        "_atom_site.label_atom_id",
        "_atom_site.label_asym_id",
        "_atom_site.auth_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.label_alt_id",
        "_atom_site.occupancy",
        "_atom_site.pdbx_PDB_model_num",
    ]
    table = block.find(tags)
    names = [tag.replace("_atom_site.", "") for tag in tags]
    atom_rows = [
        {name: str(value) for name, value in zip(names, row)}
        for row in table
    ]
    chem_tags = [
        "_chem_comp.id",
        "_chem_comp.mon_nstd_parent_comp_id",
    ]
    chem_table = block.find(chem_tags)
    chem_names = [
        tag.replace("_chem_comp.", "") for tag in chem_tags
    ]
    component_parents = _component_parent_mapping(
        [
            {
                name: str(value)
                for name, value in zip(chem_names, row)
            }
            for row in chem_table
        ]
    )
    for component in {
        _strip_cif_quotes(row["label_comp_id"]).upper()
        for row in atom_rows
        if row.get("label_comp_id")
    }:
        if component in component_parents:
            continue
        residue_info = gemmi_module.find_tabulated_residue(component)
        one_letter = str(
            getattr(residue_info, "one_letter_code", "")
        ).strip().upper().replace("T", "U")
        residue_kind = getattr(residue_info, "kind", None)
        nucleic_acid_kinds = {
            gemmi_module.ResidueKind.RNA,
            gemmi_module.ResidueKind.DNA,
        }
        if (
            residue_kind in nucleic_acid_kinds
            and one_letter in RNA_BASE_TO_ID
        ):
            component_parents[component] = one_letter
    return _assemble_cif_rna_atoms(
        atom_rows,
        chain_id,
        component_parents,
        expected_sequence=expected_sequence,
    )


def _strip_cif_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
