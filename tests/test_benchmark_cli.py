import json
import subprocess
import sys


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
    expected = ["candidates.csv", "motifs.csv", "summary.json", "run_manifest.json"]
    assert all((tmp_path / name).is_file() for name in expected)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert {row["method"] for row in summary["methods"]} == {"uniform", "markov"}


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
