import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts.train_and_evaluate_3d import (
    build_pipeline_commands,
    validate_training_artifacts,
)


def test_release_pipeline_runs_external_and_split_all_atom_gates(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "sequences_csv": "/data root/train_sequences.v2.csv",
                    "cif_dir": "/data root/PDB_RNA",
                    "max_sequence_length": 1024,
                    "external_holdout": {
                        "sequences_csv": (
                            str(
                                Path("/external data")
                                / "validation_sequences.csv"
                            )
                        ),
                        "manifest_path": "checkpoints/holdout.json",
                        "kmer_size": 8,
                        "jaccard_threshold": 0.8,
                    },
                },
                "trainer": {
                    "checkpoint_dir": "checkpoints",
                    "sequence_split": {
                        "manifest_path": "checkpoints/split_manifest.json",
                        "kmer_size": 8,
                        "jaccard_threshold": 0.8,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    train, external, all_atom = build_pipeline_commands(
        "/external data",
        config,
        "checkpoints/rna_3d_best.pt",
        "outputs/external.json",
        "outputs/all_atom.json",
        "cuda:1",
    )

    assert train[-2:] == ["--config", str(config)]
    assert str(Path("/external data") / "validation_sequences.csv") in external
    assert "--labels-csv" in external
    assert "--split-manifest" in all_atom
    assert "checkpoints/split_manifest.json" in all_atom
    assert "/data root/PDB_RNA" in all_atom
    assert "--require-max-torsion-mae-deg" in all_atom
    assert "--require-max-sugar-pucker-mae-deg" in all_atom
    assert "--require-max-base-orientation-mae-deg" in all_atom
    assert "--require-min-target-pass-fraction" in external
    assert "--require-min-target-pass-fraction" in all_atom
    assert external[-1] == "outputs/external.json"
    assert all_atom[-1] == "outputs/all_atom.json"


def test_release_pipeline_rejects_config_without_split_provenance(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "sequences_csv": "sequences.csv",
                    "cif_dir": "PDB_RNA",
                    "max_sequence_length": 512,
                },
                "trainer": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest_path"):
        build_pipeline_commands(
            Path("data"),
            config,
            "best.pt",
            "external.json",
            "all_atom.json",
            "cpu",
        )


def test_release_pipeline_rejects_missing_external_holdout(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "sequences_csv": "train.csv",
                    "cif_dir": "cif",
                    "max_sequence_length": 512,
                },
                "trainer": {
                    "checkpoint_dir": "checkpoints",
                    "sequence_split": {
                        "manifest_path": "split.json",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="external_holdout"):
        build_pipeline_commands(
            tmp_path,
            config,
            "checkpoints/rna_3d_best.pt",
            "external.json",
            "all.json",
            "cpu",
        )


def _write_release_artifacts(tmp_path: Path, *, semantics: str = "same"):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = {
        "format_version": 5,
        "architecture_version": "rna_ipa_internal_coords_v10",
        "training_semantics": semantics,
        "epoch": 2,
    }
    split_path = checkpoint_dir / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "format_version": 3,
                "candidate_strategy": "exhaustive_length_bounded",
                "audit": {"cross_split_audit_exhaustive": True},
            }
        ),
        encoding="utf-8",
    )
    holdout_path = checkpoint_dir / "holdout.json"
    holdout_path.write_text(
        json.dumps(
            {
                "format_version": 2,
                "parameters": {
                    "candidate_strategy": "exhaustive_cross_product"
                },
                "audit": {"cross_pair_audit_exhaustive": True},
            }
        ),
        encoding="utf-8",
    )
    checkpoint["training_provenance"] = {
        "split_manifest_sha256": hashlib.sha256(
            split_path.read_bytes()
        ).hexdigest(),
        "holdout_manifest_sha256": hashlib.sha256(
            holdout_path.read_bytes()
        ).hexdigest(),
    }
    torch.save(checkpoint, checkpoint_dir / "rna_3d_best.pt")
    checkpoint["epoch"] = 3
    torch.save(checkpoint, checkpoint_dir / "rna_3d_last.pt")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "external_holdout": {
                        "manifest_path": "checkpoints/holdout.json"
                    }
                },
                "trainer": {
                    "checkpoint_dir": "checkpoints",
                    "sequence_split": {
                        "manifest_path": "checkpoints/split.json"
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return config, checkpoint_dir


def test_training_artifacts_bind_best_last_and_manifests(tmp_path):
    config, checkpoint_dir = _write_release_artifacts(tmp_path)

    provenance = validate_training_artifacts(
        config, checkpoint_dir / "rna_3d_best.pt", tmp_path
    )

    assert provenance == {
        "architecture_version": "rna_ipa_internal_coords_v10",
        "training_semantics": "same",
        "best_epoch": 2,
        "last_epoch": 3,
    }


def test_training_artifacts_reject_mismatched_checkpoint_semantics(tmp_path):
    config, checkpoint_dir = _write_release_artifacts(tmp_path)
    last = torch.load(
        checkpoint_dir / "rna_3d_last.pt",
        map_location="cpu",
        weights_only=False,
    )
    last["training_semantics"] = "different"
    torch.save(last, checkpoint_dir / "rna_3d_last.pt")

    with pytest.raises(ValueError, match="do not share"):
        validate_training_artifacts(
            config, checkpoint_dir / "rna_3d_best.pt", tmp_path
        )


def test_training_artifacts_reject_non_exhaustive_manifest(tmp_path):
    config, checkpoint_dir = _write_release_artifacts(tmp_path)
    split_path = checkpoint_dir / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["audit"]["cross_split_audit_exhaustive"] = False
    split_path.write_text(json.dumps(split), encoding="utf-8")

    with pytest.raises(ValueError, match="strict exhaustive"):
        validate_training_artifacts(
            config, checkpoint_dir / "rna_3d_best.pt", tmp_path
        )


def test_training_artifacts_reject_manifest_changed_after_checkpoint(tmp_path):
    config, checkpoint_dir = _write_release_artifacts(tmp_path)
    split_path = checkpoint_dir / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["tampered_after_training"] = True
    split_path.write_text(json.dumps(split), encoding="utf-8")

    with pytest.raises(ValueError, match="current split/holdout"):
        validate_training_artifacts(
            config, checkpoint_dir / "rna_3d_best.pt", tmp_path
        )


def test_release_pipeline_rejects_different_holdout_file(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "sequences_csv": "train.csv",
                    "cif_dir": "cif",
                    "max_sequence_length": 512,
                    "external_holdout": {
                        "sequences_csv": str(tmp_path / "other.csv"),
                        "manifest_path": "holdout.json",
                    },
                },
                    "trainer": {
                        "checkpoint_dir": "checkpoints",
                        "sequence_split": {
                        "manifest_path": "split.json",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="same file"):
        build_pipeline_commands(
            tmp_path,
            config,
            "checkpoints/rna_3d_best.pt",
            "external.json",
            "all.json",
            "cpu",
        )


@pytest.mark.parametrize(
    ("holdout_field", "holdout_value", "message"),
    [
        ("kmer_size", 7, "kmer_size"),
        ("jaccard_threshold", 0.9, "jaccard_threshold"),
        ("manifest_path", "split.json", "must be distinct"),
    ],
)
def test_release_pipeline_rejects_inconsistent_holdout_policy(
    tmp_path, holdout_field, holdout_value, message
):
    holdout = {
        "sequences_csv": str(tmp_path / "validation_sequences.csv"),
        "manifest_path": "holdout.json",
        "kmer_size": 8,
        "jaccard_threshold": 0.8,
    }
    holdout[holdout_field] = holdout_value
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "sequences_csv": "train.csv",
                    "cif_dir": "cif",
                    "max_sequence_length": 512,
                    "external_holdout": holdout,
                },
                "trainer": {
                    "checkpoint_dir": "checkpoints",
                    "sequence_split": {
                        "manifest_path": "split.json",
                        "kmer_size": 8,
                        "jaccard_threshold": 0.8,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        build_pipeline_commands(
            tmp_path,
            config,
            "checkpoints/rna_3d_best.pt",
            "external.json",
            "all.json",
            "cpu",
        )
