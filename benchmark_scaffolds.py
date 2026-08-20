from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import random
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from rna_scaffold.evaluation import (
    CandidateMetric,
    maximum_homopolymer_run,
    nearest_training_similarity,
    paired_bootstrap_by_motif,
    summarize_candidates,
    within_group_diversity,
)
from rna_scaffold.generate import BASES, RnaTrainingPrior
from rna_scaffold.utils import validate_rna_sequence
from rna_scaffold.validators.rnafold import RnafoldResult, run_rnafold

ARTIFACT_NAMES = (
    "candidate_results.csv",
    "motif_summary.csv",
    "model_summary.csv",
    "paired_bootstrap.json",
    "run_manifest.json",
    "benchmark_report.md",
)
BOOTSTRAP_METRICS = (
    "valid_rate",
    "motif_preservation_rate",
    "unique_rate",
    "mean_normalized_edit_diversity",
    "mean_kmer_diversity",
    "mean_nearest_training_kmer_similarity",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_sha256(path: Path) -> str:
    if path.is_file():
        return _sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(f"benchmark data does not exist: {path}")
    digest = hashlib.sha256()
    for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(child)))
    return digest.hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            newline="",
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("benchmark CSV artifacts require at least one row")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _uniform(motif: str, length: int, rng: random.Random) -> tuple[str, int]:
    scaffold_length = length - len(motif)
    start = rng.randint(1, scaffold_length - 1)
    left = "".join(rng.choice(BASES) for _ in range(start))
    right = "".join(rng.choice(BASES) for _ in range(scaffold_length - start))
    return left + motif + right, start


def _markov(
    motif: str,
    length: int,
    prior: RnaTrainingPrior,
    rng: random.Random,
    order: int,
) -> tuple[str, int]:
    scaffold_length = length - len(motif)
    start = rng.randint(1, scaffold_length - 1)
    left = prior.sample_sequence(start, rng, order=order)
    right = prior.sample_sequence(scaffold_length - start, rng, order=order)
    return left + motif + right, start


def _normalise_method_specs(config: dict[str, Any], smoke_test: bool) -> list[dict[str, Any]]:
    raw_methods: list[Any]
    if smoke_test:
        raw_methods = [
            {"name": "uniform", "kind": "uniform"},
            {"name": "markov1", "kind": "markov1"},
            {"name": "markov2", "kind": "markov2"},
        ]
    else:
        raw_methods = list(config.get("methods") or [])
    if not raw_methods:
        raise ValueError("benchmark requires at least one method")

    specs: list[dict[str, Any]] = []
    for raw_method in raw_methods:
        method = {"name": raw_method, "kind": raw_method} if isinstance(raw_method, str) else dict(raw_method)
        method.setdefault("kind", method.get("name"))
        method.setdefault("primary", False)
        if not method.get("name"):
            raise ValueError("every benchmark method requires a name")
        if method["kind"] not in {"uniform", "markov1", "markov2", "checkpoint"}:
            raise ValueError(f"unknown benchmark method kind: {method['kind']}")
        if method["kind"] == "checkpoint":
            if not method.get("checkpoint"):
                raise ValueError(f"checkpoint method {method['name']!r} requires checkpoint")
            checkpoint = Path(str(method["checkpoint"]))
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"checkpoint for benchmark method {method['name']!r} does not exist: {checkpoint}"
                )
        specs.append(method)

    names = [str(method["name"]) for method in specs]
    if len(names) != len(set(names)):
        raise ValueError("benchmark method names must be unique")
    if sum(bool(method["primary"]) for method in specs) > 1:
        raise ValueError("at most one benchmark method may be primary")
    return specs


def _normalise_motifs(config: dict[str, Any]) -> list[dict[str, str]]:
    motifs = [
        {"id": str(row["id"]), "sequence": str(row["sequence"]).strip().upper().replace("T", "U")}
        for row in (config.get("motifs") or [])
    ]
    if not motifs:
        raise ValueError("benchmark requires at least one motif")
    if len({row["id"] for row in motifs}) != len(motifs):
        raise ValueError("benchmark motif IDs must be unique")
    for row in motifs:
        if not validate_rna_sequence(row["sequence"]):
            raise ValueError(f"invalid benchmark motif {row['id']!r}")
    return motifs


def _configured_primary(config: dict[str, Any]) -> str | None:
    for raw_method in config.get("methods") or []:
        if isinstance(raw_method, dict) and raw_method.get("primary"):
            return str(raw_method.get("name"))
    return None


def _data_hashes(config: dict[str, Any]) -> dict[str, str]:
    paths: list[Path] = []
    for key in ("training_data", "cluster_manifest"):
        if config.get(key):
            paths.append(Path(str(config[key])))
    configured_data = config.get("data_files") or {}
    values = configured_data.values() if isinstance(configured_data, dict) else configured_data
    paths.extend(Path(str(value)) for value in values)
    return {str(path): _path_sha256(path) for path in paths}


def _candidate_metric(
    motif_id: str,
    motif: str,
    sequence: str,
    start: int | None,
    training_sequences: list[str],
    normalized_edit_diversity: float,
    kmer_diversity: float,
    runtime_seconds: float,
    peak_cuda_memory_bytes: int | None,
    failure: str | None,
    rejection_reason: str | None,
    rnafold: RnafoldResult | None,
) -> CandidateMetric:
    preserved = start is not None and sequence[start : start + len(motif)] == motif
    valid = bool(sequence) and validate_rna_sequence(sequence) and preserved and failure is None
    gc_fraction = (sequence.count("G") + sequence.count("C")) / len(sequence) if sequence else 0.0
    if failure is None and not valid:
        failure = "invalid_sequence"
    return CandidateMetric(
        motif_id=motif_id,
        sequence=sequence,
        valid=valid,
        motif_preserved=preserved,
        total_length=len(sequence),
        gc_fraction=gc_fraction,
        failure=failure,
        normalized_edit_diversity=normalized_edit_diversity,
        kmer_diversity=kmer_diversity,
        nearest_training_kmer_similarity=(
            nearest_training_similarity(sequence, training_sequences, k=5) if sequence else None
        ),
        max_homopolymer_run=maximum_homopolymer_run(sequence),
        runtime_seconds=runtime_seconds,
        peak_cuda_memory_bytes=peak_cuda_memory_bytes,
        rejection_reason=rejection_reason,
        rnafold_status=rnafold.status if rnafold else None,
        rnafold_dot_bracket=rnafold.dot_bracket if rnafold else None,
        rnafold_mfe_kcal_mol=rnafold.mfe_kcal_mol if rnafold else None,
        rnafold_paired_fraction=rnafold.paired_fraction if rnafold else None,
        rnafold_motif_paired_fraction=rnafold.motif_paired_fraction if rnafold else None,
        rnafold_runtime_seconds=rnafold.runtime_seconds if rnafold else None,
        rnafold_version=rnafold.version if rnafold else None,
        rnafold_error=rnafold.error if rnafold else None,
    )


def _generate_method_motif(
    method: dict[str, Any],
    motif: str,
    length: int,
    budget: int,
    seed: int,
    prior: RnaTrainingPrior,
) -> tuple[list[dict[str, Any]], dict[str, int], int | None]:
    kind = str(method["kind"])
    rng = random.Random(seed)
    if kind in {"uniform", "markov1", "markov2"}:
        generated = []
        for _ in range(budget):
            sequence, start = (
                _uniform(motif, length, rng)
                if kind == "uniform"
                else _markov(motif, length, prior, rng, order=1 if kind == "markov1" else 2)
            )
            generated.append({"sequence": sequence, "motif_start": start, "failure": None, "rejection_reason": None})
        return generated, {}, None

    from rna_scaffold.generate import GenerationSettings, generate_candidates

    generation = dict(method.get("generation") or {})
    generation.update(num_candidates=budget, seed=seed)
    device = str(method.get("device", "cpu"))
    using_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if using_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    learned_candidates = generate_candidates(
        method["checkpoint"],
        motif,
        GenerationSettings(**generation),
        device=device,
    )
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if using_cuda else None
    generated = [
        {
            "sequence": candidate.full_sequence,
            "motif_start": candidate.motif_start,
            "failure": None if candidate.valid else candidate.status,
            "rejection_reason": None,
        }
        for candidate in learned_candidates
    ]
    rejection_counts = dict(learned_candidates.audit.rejected)
    shortfall = budget - len(generated)
    if shortfall > 0:
        rejection_reason = (
            ";".join(f"{reason}:{count}" for reason, count in sorted(rejection_counts.items()))
            or "candidate_budget_exhausted"
        )
        generated.extend(
            {
                "sequence": "",
                "motif_start": None,
                "failure": "generation_shortfall",
                "rejection_reason": rejection_reason,
            }
            for _ in range(shortfall)
        )
    return generated, rejection_counts, peak_memory


def _rnafold_results(
    generated: list[dict[str, Any]],
    motif_length: int,
    config: dict[str, Any],
    budget: int,
) -> list[RnafoldResult | None]:
    rnafold_config = dict(config.get("rnafold") or {})
    if not rnafold_config.get("enabled", False):
        return [None] * len(generated)
    structure_budget = int(rnafold_config.get("candidate_budget_per_motif", budget))
    if structure_budget < 0:
        raise ValueError("rnafold candidate budget must be non-negative")
    results: list[RnafoldResult | None] = []
    used = 0
    for candidate in generated:
        start = candidate["motif_start"]
        if not candidate["sequence"] or start is None or used >= structure_budget:
            results.append(None)
            continue
        results.append(
            run_rnafold(
                candidate["sequence"],
                motif_start=start,
                motif_end=start + motif_length,
                executable=rnafold_config.get("executable", "RNAfold"),
                timeout_seconds=float(rnafold_config.get("timeout_seconds", 30.0)),
            )
        )
        used += 1
    return results


def _paired_bootstrap_artifact(
    motif_summaries: list[dict[str, Any]],
    methods: list[str],
    seed: int,
    samples: int,
) -> dict[str, Any]:
    comparison_rows: list[dict[str, Any]] = []
    by_method = {
        method: {row["motif_id"]: row for row in motif_summaries if row["method"] == method} for method in methods
    }
    for left_method, right_method in combinations(methods, 2):
        for metric in BOOTSTRAP_METRICS:
            left_values = {
                motif_id: float(row[metric])
                for motif_id, row in by_method[left_method].items()
                if row.get(metric) is not None
            }
            right_values = {
                motif_id: float(row[metric])
                for motif_id, row in by_method[right_method].items()
                if row.get(metric) is not None
            }
            if not set(left_values) & set(right_values):
                continue
            common_ids, difference = paired_bootstrap_by_motif(
                left_values,
                right_values,
                seed=seed,
                samples=samples,
            )
            comparison_rows.append(
                {
                    "left_method": left_method,
                    "right_method": right_method,
                    "metric": metric,
                    "common_motif_ids": list(common_ids),
                    **asdict(difference),
                }
            )
    return {"seed": seed, "samples": samples, "comparisons": comparison_rows}


def _report(
    model_summaries: list[dict[str, Any]],
    primary_name: str | None,
    budget: int,
    seed: int,
) -> str:
    primary_line = f"Primary model: `{primary_name}`" if primary_name else "Primary model: none configured"
    lines = [
        "# RNA Scaffold Benchmark",
        "",
        primary_line,
        "",
        f"Candidate budget per motif: {budget}; deterministic seed: {seed}.",
        "",
        ("RNAfold-derived MFE and pairing metrics are a plausibility proxy only; they are not proof of function."),
        "",
        "| Method | Primary | Status | Valid rate | Unique rate | Edit diversity | Failures | Runtime (s) |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in model_summaries:
        method = f"**{row['method']}**" if row["primary"] else row["method"]
        lines.append(
            "| {method} | {primary} | {status} | {valid:.4f} | {unique:.4f} | "
            "{diversity:.4f} | {failures} | {runtime:.4f} |".format(
                method=method,
                primary="yes" if row["primary"] else "no",
                status=row["status"],
                valid=row["valid_rate"],
                unique=row["unique_rate"],
                diversity=row["mean_normalized_edit_diversity"],
                failures=row["failure_count"],
                runtime=row["runtime_seconds"],
            )
        )
    return "\n".join(lines) + "\n"


def _software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "pyyaml": yaml.__version__,
    }


def run_benchmark(config_path: Path, output_dir: Path, smoke_test: bool = False) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("benchmark configuration must be a mapping")
    seed = int(config.get("seed", 42))
    budget = 4 if smoke_test else int(config.get("candidate_count", 16))
    if budget <= 0:
        raise ValueError("candidate_count must be positive")
    bootstrap_samples = 100 if smoke_test else int(config.get("bootstrap_samples", 10000))
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    method_specs = _normalise_method_specs(config, smoke_test)
    motifs = _normalise_motifs(config)
    primary_name = _configured_primary(config)
    if not smoke_test:
        primary_name = next(
            (str(method["name"]) for method in method_specs if method["primary"]),
            primary_name,
        )
    data_sha256 = {} if smoke_test else _data_hashes(config)
    checkpoint_sha256 = {
        str(method["name"]): _sha256(Path(str(method["checkpoint"])))
        for method in method_specs
        if method["kind"] == "checkpoint"
    }
    training_sequences: list[str] = []
    training_data = config.get("training_data")
    if training_data and not smoke_test:
        from rna_scaffold.data import load_sequences

        training_sequences = load_sequences(training_data)
    prior = RnaTrainingPrior.from_sequences(training_sequences) if training_sequences else RnaTrainingPrior.empty()

    started = time.perf_counter()
    candidate_rows: list[dict[str, Any]] = []
    motif_summaries: list[dict[str, Any]] = []
    model_summaries: list[dict[str, Any]] = []
    manifest_methods: list[dict[str, Any]] = []

    for method in method_specs:
        method_name = str(method["name"])
        method_started = time.perf_counter()
        method_metrics: list[CandidateMetric] = []
        method_rejections: Counter[str] = Counter()
        motif_seeds: dict[str, int] = {}
        method_partial = False
        for motif_index, motif_row in enumerate(motifs):
            motif_id = motif_row["id"]
            motif = motif_row["sequence"]
            motif_seed = seed + motif_index * 1000
            motif_seeds[motif_id] = motif_seed
            length = max(len(motif) + 8, 24)
            generation_started = time.perf_counter()
            generated, rejection_counts, peak_memory = _generate_method_motif(
                method,
                motif,
                length,
                budget,
                motif_seed,
                prior,
            )
            generation_runtime = time.perf_counter() - generation_started
            method_rejections.update(rejection_counts)
            method_partial = method_partial or any(row["failure"] == "generation_shortfall" for row in generated)
            actual_sequences = [row["sequence"] for row in generated if row["sequence"]]
            edit_scores, kmer_scores = within_group_diversity(actual_sequences, k=5)
            actual_index = 0
            rnafold_results = _rnafold_results(generated, len(motif), config, budget)
            motif_metrics: list[CandidateMetric] = []
            per_candidate_runtime = generation_runtime / budget
            for candidate_index, (generated_row, rnafold_result) in enumerate(zip(generated, rnafold_results)):
                has_sequence = bool(generated_row["sequence"])
                edit_diversity = edit_scores[actual_index] if has_sequence else 0.0
                kmer_diversity = kmer_scores[actual_index] if has_sequence else 0.0
                if has_sequence:
                    actual_index += 1
                metric = _candidate_metric(
                    motif_id,
                    motif,
                    generated_row["sequence"],
                    generated_row["motif_start"],
                    training_sequences,
                    edit_diversity,
                    kmer_diversity,
                    per_candidate_runtime,
                    peak_memory,
                    generated_row["failure"],
                    generated_row["rejection_reason"],
                    rnafold_result,
                )
                method_metrics.append(metric)
                motif_metrics.append(metric)
                candidate_rows.append(
                    {
                        "method": method_name,
                        "primary": bool(method["primary"]),
                        "motif_id": motif_id,
                        "candidate_index": candidate_index,
                        "motif_start": generated_row["motif_start"],
                        "motif_end": (
                            generated_row["motif_start"] + len(motif)
                            if generated_row["motif_start"] is not None
                            else None
                        ),
                        "seed": motif_seed,
                        **asdict(metric),
                    }
                )
            motif_summaries.append(
                {
                    "method": method_name,
                    "primary": bool(method["primary"]),
                    "motif_id": motif_id,
                    "seed": motif_seed,
                    "rejection_reasons": json.dumps(rejection_counts, sort_keys=True),
                    **asdict(summarize_candidates(motif_metrics)),
                }
            )
        method_runtime = time.perf_counter() - method_started
        status = "partial" if method_partial else "ok"
        method_summary = {
            "method": method_name,
            "kind": str(method["kind"]),
            "primary": bool(method["primary"]),
            "status": status,
            "seed": seed,
            "rejection_reasons": json.dumps(dict(sorted(method_rejections.items())), sort_keys=True),
            **asdict(summarize_candidates(method_metrics)),
        }
        method_summary["runtime_seconds"] = method_runtime
        model_summaries.append(method_summary)
        manifest_methods.append(
            {
                "name": method_name,
                "kind": str(method["kind"]),
                "primary": bool(method["primary"]),
                "status": status,
                "seed": seed,
                "motif_seeds": motif_seeds,
                "runtime_seconds": method_runtime,
                "checkpoint_sha256": checkpoint_sha256.get(method_name),
                "rejection_reasons": dict(sorted(method_rejections.items())),
            }
        )

    paired_artifact = _paired_bootstrap_artifact(
        motif_summaries,
        [str(method["name"]) for method in method_specs],
        seed,
        bootstrap_samples,
    )
    report = _report(model_summaries, primary_name, budget, seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_content = {
        "candidate_results.csv": _csv_text(candidate_rows),
        "motif_summary.csv": _csv_text(motif_summaries),
        "model_summary.csv": _csv_text(model_summaries),
        "paired_bootstrap.json": json.dumps(paired_artifact, indent=2, sort_keys=True) + "\n",
        "benchmark_report.md": report,
    }
    for name, content in artifact_content.items():
        _atomic_write_text(output_dir / name, content)

    runtime_seconds = time.perf_counter() - started
    software_versions = _software_versions()
    rnafold_versions = sorted({str(row["rnafold_version"]) for row in candidate_rows if row.get("rnafold_version")})
    if rnafold_versions:
        software_versions["rnafold"] = ", ".join(rnafold_versions)
    elif (config.get("rnafold") or {}).get("enabled", False):
        software_versions["rnafold"] = "unavailable"
    manifest = {
        "status": "completed",
        "configuration": config,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "data_sha256": data_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "seed": seed,
        "candidate_budget_per_motif": budget,
        "rnafold_candidate_budget_per_motif": int(
            (config.get("rnafold") or {}).get("candidate_budget_per_motif", budget)
        ),
        "bootstrap_samples": bootstrap_samples,
        "smoke_test": smoke_test,
        "command": list(sys.argv),
        "software_versions": software_versions,
        "methods": manifest_methods,
        "runtime_seconds": runtime_seconds,
        "artifact_sha256": {name: _sha256(output_dir / name) for name in artifact_content},
    }
    _atomic_write_text(
        output_dir / "run_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RNA scaffold generators fairly.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir or config.get("output_dir", "outputs/scaffold_benchmark"))
    run_benchmark(config_path, output_dir, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
