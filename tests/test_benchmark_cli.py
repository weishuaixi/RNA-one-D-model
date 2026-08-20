import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import benchmark_scaffolds
from rna_scaffold.generate import CandidateGenerationAudit
from rna_scaffold.validators.rnafold import RnafoldResult


def test_benchmark_smoke_writes_reproducible_artifacts(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "benchmark_scaffolds.py",
            "--config",
            "configs/benchmark_scaffolds.yaml",
            "--smoke-test",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    expected = {
        "candidate_results.csv",
        "motif_summary.csv",
        "model_summary.csv",
        "paired_bootstrap.json",
        "run_manifest.json",
        "benchmark_report.md",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected

    with (tmp_path / "candidate_results.csv").open(encoding="utf-8", newline="") as handle:
        candidates = list(csv.DictReader(handle))
    assert {row["method"] for row in candidates} == {"uniform", "markov1", "markov2"}
    assert all(
        len({row["seed"] for row in candidates if row["motif_id"] == motif_id}) == 1
        for motif_id in {row["motif_id"] for row in candidates}
    )
    assert {
        (row["method"], row["motif_id"]): sum(
            candidate["method"] == row["method"] and candidate["motif_id"] == row["motif_id"]
            for candidate in candidates
        )
        for row in candidates
    } == {(method, motif): 4 for method in ("uniform", "markov1", "markov2") for motif in ("motif_gugg", "motif_augg")}
    assert {
        "normalized_edit_diversity",
        "kmer_diversity",
        "nearest_training_kmer_similarity",
        "max_homopolymer_run",
        "runtime_seconds",
        "peak_cuda_memory_bytes",
        "rnafold_status",
        "rnafold_mfe_kcal_mol",
        "rnafold_paired_fraction",
        "rnafold_motif_paired_fraction",
    } <= set(candidates[0])

    paired = json.loads((tmp_path / "paired_bootstrap.json").read_text(encoding="utf-8"))
    assert paired["comparisons"]
    assert all(comparison["common_motif_ids"] == ["motif_augg", "motif_gugg"] for comparison in paired["comparisons"])

    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["seed"] == 42
    assert manifest["configuration"]["candidate_count"] == 16
    assert (
        manifest["config_sha256"] == hashlib.sha256(Path("configs/benchmark_scaffolds.yaml").read_bytes()).hexdigest()
    )
    assert manifest["command"][-3:] == ["--output-dir", str(tmp_path), "--smoke-test"] or manifest["command"][-3:] == [
        "--smoke-test",
        "--output-dir",
        str(tmp_path),
    ]
    assert {"python", "numpy", "torch", "pyyaml"} <= set(manifest["software_versions"])
    assert set(manifest["artifact_sha256"]) == expected - {"run_manifest.json"}
    assert {method["status"] for method in manifest["methods"]} == {"ok"}
    assert manifest["runtime_seconds"] >= 0

    report = (tmp_path / "benchmark_report.md").read_text(encoding="utf-8")
    assert "plausibility proxy" in report
    assert "not proof of function" in report


def test_benchmark_marks_configured_primary_and_hashes_training_data(tmp_path):
    training_data = tmp_path / "train.fasta"
    training_data.write_text(">train\nAAAAGCGGUUUU\n", encoding="utf-8")
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "seed": 9,
                "candidate_count": 2,
                "bootstrap_samples": 100,
                "motifs": [
                    {"id": "m1", "sequence": "GCGG"},
                    {"id": "m2", "sequence": "AUGC"},
                ],
                "methods": [
                    {"name": "control", "kind": "uniform"},
                    {"name": "primary_model", "kind": "markov2", "primary": True},
                ],
                "training_data": str(training_data),
                "rnafold": {"enabled": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            "benchmark_scaffolds.py",
            "--config",
            str(config),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    with (output / "model_summary.csv").open(encoding="utf-8", newline="") as handle:
        summaries = {row["method"]: row for row in csv.DictReader(handle)}
    assert summaries["primary_model"]["primary"].lower() == "true"
    assert summaries["control"]["primary"].lower() == "false"

    with (output / "candidate_results.csv").open(encoding="utf-8", newline="") as handle:
        candidates = list(csv.DictReader(handle))
    assert all(row["nearest_training_kmer_similarity"] for row in candidates)
    assert all(int(row["max_homopolymer_run"]) >= 1 for row in candidates)

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_sha256"] == {str(training_data): hashlib.sha256(training_data.read_bytes()).hexdigest()}
    assert manifest["checkpoint_sha256"] == {}
    assert {row["seed"] for row in manifest["methods"]} == {9}
    assert "Primary model: `primary_model`" in (output / "benchmark_report.md").read_text(encoding="utf-8")


def test_learned_benchmark_method_requires_checkpoint(tmp_path):
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        "seed: 42\n"
        "candidate_count: 2\n"
        "motifs:\n"
        "  - {id: m1, sequence: GCGG}\n"
        "methods:\n"
        "  - {name: complete, kind: checkpoint}\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "benchmark_scaffolds.py",
            "--config",
            str(config),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "checkpoint" in completed.stderr.lower()


def test_missing_checkpoint_fails_before_writing_any_fallback_artifacts(tmp_path):
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "candidate_count": 2,
                "motifs": [{"id": "m1", "sequence": "GCGG"}],
                "methods": [
                    {
                        "name": "primary_model",
                        "kind": "checkpoint",
                        "checkpoint": str(tmp_path / "missing.ckpt"),
                        "primary": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            "benchmark_scaffolds.py",
            "--config",
            str(config),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "checkpoint" in completed.stderr.lower()
    assert "does not exist" in completed.stderr.lower()
    assert not output.exists()


def test_checkpoint_shortfall_is_explicit_and_retains_rejection_reasons(monkeypatch, tmp_path):
    checkpoint = tmp_path / "tiny.ckpt"
    checkpoint.write_bytes(b"v2 checkpoint placeholder")
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "seed": 5,
                "candidate_count": 2,
                "bootstrap_samples": 10,
                "motifs": [{"id": "m1", "sequence": "GCGG"}],
                "methods": [
                    {
                        "name": "primary_model",
                        "kind": "checkpoint",
                        "checkpoint": str(checkpoint),
                        "primary": True,
                    }
                ],
                "rnafold": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )

    class ShortBatch(list):
        audit = CandidateGenerationAudit(
            requested=2,
            max_attempts=4,
            attempted=4,
            accepted=1,
            rejected={"duplicate": 3},
        )

    monkeypatch.setattr(
        "rna_scaffold.generate.generate_candidates",
        lambda *args, **kwargs: ShortBatch(
            [
                SimpleNamespace(
                    full_sequence="AAGCGGUU",
                    motif_start=2,
                    valid=True,
                    status="shortfall",
                )
            ]
        ),
    )
    output = tmp_path / "out"

    benchmark_scaffolds.run_benchmark(config, output)

    with (output / "candidate_results.csv").open(encoding="utf-8", newline="") as handle:
        candidates = list(csv.DictReader(handle))
    assert len(candidates) == 2
    assert candidates[0]["failure"] == ""
    assert candidates[1]["failure"] == "generation_shortfall"
    assert candidates[1]["rejection_reason"] == "duplicate:3"
    with (output / "model_summary.csv").open(encoding="utf-8", newline="") as handle:
        summary = next(csv.DictReader(handle))
    assert summary["status"] == "partial"
    assert summary["failure_count"] == "1"
    assert float(summary["unique_rate"]) == pytest.approx(0.5)
    assert float(summary["mean_length"]) == pytest.approx(8.0)
    assert json.loads(summary["rejection_reasons"]) == {"duplicate": 3}


def test_optional_rnafold_enrichment_records_status_and_proxy_metrics(monkeypatch, tmp_path):
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "seed": 3,
                "candidate_count": 1,
                "bootstrap_samples": 10,
                "motifs": [{"id": "m1", "sequence": "GCGG"}],
                "methods": [{"name": "uniform", "kind": "uniform"}],
                "rnafold": {
                    "enabled": True,
                    "candidate_budget_per_motif": 1,
                    "executable": "fake-rnafold",
                },
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_rnafold(sequence, motif_start, motif_end, executable, timeout_seconds):
        calls.append((sequence, motif_start, motif_end, executable, timeout_seconds))
        return RnafoldResult("ok", "." * len(sequence), -3.5, 0.5, 0.25, 0.01, "RNAfold 2.7", None)

    monkeypatch.setattr(benchmark_scaffolds, "run_rnafold", fake_rnafold)
    output = tmp_path / "out"

    benchmark_scaffolds.run_benchmark(config, output)

    assert len(calls) == 1
    with (output / "candidate_results.csv").open(encoding="utf-8", newline="") as handle:
        candidate = next(csv.DictReader(handle))
    assert candidate["rnafold_status"] == "ok"
    assert float(candidate["rnafold_mfe_kcal_mol"]) == pytest.approx(-3.5)
    assert float(candidate["rnafold_paired_fraction"]) == pytest.approx(0.5)
    assert float(candidate["rnafold_motif_paired_fraction"]) == pytest.approx(0.25)
    assert candidate["rnafold_version"] == "RNAfold 2.7"
    assert float(candidate["rnafold_runtime_seconds"]) == pytest.approx(0.01)
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["software_versions"]["rnafold"] == "RNAfold 2.7"


def test_atomic_writer_preserves_existing_artifact_when_replace_fails(monkeypatch, tmp_path):
    destination = tmp_path / "model_summary.csv"
    destination.write_text("old complete artifact\n", encoding="utf-8")
    monkeypatch.setattr(
        benchmark_scaffolds.os,
        "replace",
        lambda source, target: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        benchmark_scaffolds._atomic_write_text(destination, "partial new artifact\n")

    assert destination.read_text(encoding="utf-8") == "old complete artifact\n"
    assert [path.name for path in tmp_path.iterdir()] == ["model_summary.csv"]
