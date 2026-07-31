from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import yaml


def _resolve_artifact(path: str | Path, repository_root: Path) -> Path:
    path = Path(path)
    return (repository_root / path).resolve() if not path.is_absolute() else path.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_training_artifacts(
    config_path: str | Path,
    checkpoint_path: str | Path,
    repository_root: str | Path,
) -> dict[str, object]:
    """Fail closed before evaluating stale or unrelated training artifacts."""
    repository_root = Path(repository_root).resolve()
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    trainer = config.get("trainer", {})
    data = config.get("data", {})
    split = trainer.get("sequence_split", {})
    holdout = data.get("external_holdout", {})
    checkpoint_dir = _resolve_artifact(
        trainer["checkpoint_dir"], repository_root
    )
    expected_best = checkpoint_dir / "rna_3d_best.pt"
    actual_best = _resolve_artifact(checkpoint_path, repository_root)
    if actual_best != expected_best:
        raise ValueError(
            f"Release checkpoint must be {expected_best}, got {actual_best}."
        )
    paths = {
        "best checkpoint": expected_best,
        "last checkpoint": checkpoint_dir / "rna_3d_last.pt",
        "split manifest": _resolve_artifact(
            split["manifest_path"], repository_root
        ),
        "holdout manifest": _resolve_artifact(
            holdout["manifest_path"], repository_root
        ),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Training did not produce required release artifacts: "
            + ", ".join(missing)
        )

    split_payload = json.loads(
        paths["split manifest"].read_text(encoding="utf-8")
    )
    split_audit = split_payload.get("audit", {})
    if (
        split_payload.get("format_version") != 3
        or split_payload.get("candidate_strategy")
        != "exhaustive_length_bounded"
        or not isinstance(split_audit, dict)
        or split_audit.get("cross_split_audit_exhaustive") is not True
    ):
        raise ValueError(
            "Split manifest is not a strict exhaustive v3 artifact."
        )
    holdout_payload = json.loads(
        paths["holdout manifest"].read_text(encoding="utf-8")
    )
    holdout_audit = holdout_payload.get("audit", {})
    holdout_parameters = holdout_payload.get("parameters", {})
    if (
        holdout_payload.get("format_version") != 2
        or not isinstance(holdout_audit, dict)
        or holdout_audit.get("cross_pair_audit_exhaustive") is not True
        or not isinstance(holdout_parameters, dict)
        or holdout_parameters.get("candidate_strategy")
        != "exhaustive_cross_product"
    ):
        raise ValueError(
            "External holdout manifest is not an exhaustive v2 artifact."
        )

    import torch

    checkpoints = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
        if name.endswith("checkpoint")
    }
    expected_provenance = {
        "split_manifest_sha256": _sha256_file(paths["split manifest"]),
        "holdout_manifest_sha256": _sha256_file(paths["holdout manifest"]),
    }
    for name, payload in checkpoints.items():
        if payload.get("format_version") != 5:
            raise ValueError(f"{name} does not use checkpoint format v5.")
        if not payload.get("training_semantics"):
            raise ValueError(f"{name} lacks training_semantics provenance.")
        if payload.get("training_provenance") != expected_provenance:
            raise ValueError(
                f"{name} was not trained with the current split/holdout manifests."
            )
    best = checkpoints["best checkpoint"]
    last = checkpoints["last checkpoint"]
    if (
        best.get("architecture_version") != last.get("architecture_version")
        or best["training_semantics"] != last["training_semantics"]
    ):
        raise ValueError(
            "Best and last checkpoints do not share training semantics."
        )
    if int(best.get("epoch", 0)) > int(last.get("epoch", 0)):
        raise ValueError("Best checkpoint epoch cannot exceed last checkpoint epoch.")
    return {
        "architecture_version": best["architecture_version"],
        "training_semantics": best["training_semantics"],
        "best_epoch": int(best["epoch"]),
        "last_epoch": int(last["epoch"]),
    }


def build_pipeline_commands(
    data_root: str | Path,
    config_path: str | Path,
    checkpoint_path: str | Path,
    external_output: str | Path,
    all_atom_output: str | Path,
    eval_device: str,
) -> list[list[str]]:
    """Build reproducible training plus external and all-atom validation commands."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data = config.get("data", {})
    trainer = config.get("trainer", {})
    split = trainer.get("sequence_split", {})
    holdout = data.get("external_holdout", {})
    required = {
        "data.sequences_csv": data.get("sequences_csv"),
        "data.cif_dir": data.get("cif_dir"),
        "trainer.sequence_split.manifest_path": split.get("manifest_path"),
        "trainer.checkpoint_dir": trainer.get("checkpoint_dir"),
        "data.external_holdout.sequences_csv": (
            holdout.get("sequences_csv")
            if isinstance(holdout, dict)
            else None
        ),
        "data.external_holdout.manifest_path": (
            holdout.get("manifest_path")
            if isinstance(holdout, dict)
            else None
        ),
        "data.max_sequence_length": data.get("max_sequence_length"),
    }
    missing = [name for name, value in required.items() if value in (None, "")]
    if missing:
        raise ValueError(
            "Full-atom release validation requires: " + ", ".join(missing)
        )

    python = sys.executable
    root = Path(data_root)
    external_sequences = root / "validation_sequences.csv"
    configured_holdout = Path(str(holdout["sequences_csv"]))
    if configured_holdout.resolve() != external_sequences.resolve():
        raise ValueError(
            "data.external_holdout.sequences_csv must be the same file "
            "used for external release evaluation: "
            f"{external_sequences}"
        )
    split_threshold = float(split.get("jaccard_threshold", 0.8))
    holdout_threshold = float(holdout.get("jaccard_threshold", 0.8))
    if not math.isclose(
        split_threshold, holdout_threshold, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "Internal split and external holdout jaccard_threshold must match."
        )
    split_kmer_size = int(split.get("kmer_size", 8))
    holdout_kmer_size = int(holdout.get("kmer_size", 8))
    if split_kmer_size != holdout_kmer_size:
        raise ValueError(
            "Internal split and external holdout kmer_size must match."
        )
    if Path(str(holdout["manifest_path"])).resolve() == Path(
        str(split["manifest_path"])
    ).resolve():
        raise ValueError(
            "External holdout and internal split manifests must be distinct."
        )
    expected_checkpoint = Path(str(trainer["checkpoint_dir"])) / "rna_3d_best.pt"
    if Path(checkpoint_path).resolve() != expected_checkpoint.resolve():
        raise ValueError(
            "checkpoint must point to trainer.checkpoint_dir/rna_3d_best.pt."
        )
    common_gates = [
        "--recycle-counts", "all",
        "--require-min-records", "10",
        "--require-min-metric-coverage", "0.9",
        "--require-min-target-pass-fraction", "0.9",
        "--require-min-lddt", "50",
        "--require-max-kabsch-rmsd", "15",
        "--require-adjacent-c1-min", "4.5",
        "--require-adjacent-c1-max", "7.0",
        "--require-max-plddt-mae", "15",
        "--require-min-plddt-correlation", "0.3",
        "--require-max-covalent-bond-rmse", "0.1",
        "--require-max-backbone-angle-rmse-deg", "5",
        "--require-max-clash-penetration-rms", "0.05",
        "--require-max-base-planarity-rms", "0.05",
        "--require-max-sugar-closure-rmse", "0.1",
        "--require-max-o3-p-bond-rmse", "0.1",
        "--require-max-recycle-c1-rmsd", "5",
    ]
    train = [python, "train_3d.py", "--config", str(config_path)]
    external = [
        python,
        "evaluate_3d.py",
        "--checkpoint", str(checkpoint_path),
        "--sequences-csv", str(external_sequences),
        "--labels-csv", str(root / "validation_labels.csv"),
        "--device", eval_device,
        "--label-model-indices", "all",
        "--min-reference-coverage", "0.9",
        *common_gates,
        "--output", str(external_output),
    ]
    all_atom = [
        python,
        "evaluate_3d.py",
        "--checkpoint", str(checkpoint_path),
        "--sequences-csv", str(data["sequences_csv"]),
        "--cif-dir", str(data["cif_dir"]),
        "--split-manifest", str(split["manifest_path"]),
        "--max-sequence-length", str(int(data["max_sequence_length"])),
        "--device", eval_device,
        *common_gates,
        "--require-max-torsion-mae-deg", "30",
        "--require-max-sugar-pucker-mae-deg", "25",
        "--require-max-base-orientation-mae-deg", "25",
        "--output", str(all_atom_output),
    ]
    return [train, external, all_atom]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train RNA v10 and require both external C1' and held-out "
            "all-atom release gates."
        )
    )
    parser.add_argument("data_root")
    parser.add_argument(
        "config", nargs="?", default="configs/train_3d_a800_full.yaml"
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="checkpoints_3d_a800_full/rna_3d_best.pt",
    )
    parser.add_argument(
        "output", nargs="?", default="outputs/validation_metrics.json"
    )
    parser.add_argument("all_atom_output", nargs="?")
    args = parser.parse_args()
    external_output = Path(args.output)
    all_atom_output = (
        Path(args.all_atom_output)
        if args.all_atom_output
        else external_output.with_name(
            external_output.stem + "_all_atom.json"
        )
    )
    commands = build_pipeline_commands(
        args.data_root,
        args.config,
        args.checkpoint,
        external_output,
        all_atom_output,
        os.environ.get("EVAL_DEVICE", "cuda:1"),
    )
    repository_root = Path(__file__).resolve().parents[1]
    subprocess.run(commands[0], cwd=repository_root, check=True)
    provenance = validate_training_artifacts(
        args.config, args.checkpoint, repository_root
    )
    print(
        "validated_training_artifacts="
        + ", ".join(f"{key}={value}" for key, value in provenance.items())
    )
    for command in commands[1:]:
        subprocess.run(command, cwd=repository_root, check=True)


if __name__ == "__main__":
    main()
