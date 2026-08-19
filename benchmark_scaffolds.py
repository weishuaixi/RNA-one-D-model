from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from rna_scaffold.evaluation import CandidateMetric, summarize_candidates
from rna_scaffold.generate import BASES, RnaTrainingPrior
from rna_scaffold.utils import validate_rna_sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _uniform(motif: str, length: int, rng: random.Random) -> tuple[str, int]:
    scaffold_length = length - len(motif)
    start = rng.randint(1, scaffold_length - 1)
    left = "".join(rng.choice(BASES) for _ in range(start))
    right = "".join(rng.choice(BASES) for _ in range(scaffold_length - start))
    return left + motif + right, start


def _markov(motif: str, length: int, prior: RnaTrainingPrior, rng: random.Random) -> tuple[str, int]:
    scaffold_length = length - len(motif)
    start = rng.randint(1, scaffold_length - 1)
    return prior.sample_sequence(start, rng) + motif + prior.sample_sequence(scaffold_length - start, rng), start


def _metric(motif_id: str, motif: str, sequence: str, start: int) -> CandidateMetric:
    preserved = sequence[start : start + len(motif)] == motif
    valid = validate_rna_sequence(sequence) and preserved
    gc = (sequence.count("G") + sequence.count("C")) / len(sequence)
    return CandidateMetric(
        motif_id, sequence, valid, preserved, len(sequence), gc, None if valid else "invalid"
    )


def run_benchmark(config_path: Path, output_dir: Path, smoke_test: bool = False) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config.get("seed", 42))
    budget = 4 if smoke_test else int(config.get("candidate_count", 16))
    raw_methods = ["uniform", "markov"] if smoke_test else list(config["methods"])
    method_specs = [
        {"name": method, "kind": method} if isinstance(method, str) else dict(method)
        for method in raw_methods
    ]
    for method in method_specs:
        method.setdefault("kind", method.get("name"))
        if not method.get("name"):
            raise ValueError("every benchmark method requires a name")
        if method["kind"] == "checkpoint" and not method.get("checkpoint"):
            raise ValueError(f"checkpoint method {method['name']!r} requires checkpoint")
    methods = [str(method["name"]) for method in method_specs]
    motifs = list(config["motifs"])
    training_data = config.get("training_data")
    if training_data:
        from rna_scaffold.data import load_sequences

        prior = RnaTrainingPrior.from_sequences(load_sequences(training_data))
    else:
        prior = RnaTrainingPrior.empty()
    started = time.time()
    candidate_rows = []
    metric_rows: dict[str, list[CandidateMetric]] = {method: [] for method in methods}
    for method_index, method_spec in enumerate(method_specs):
        method = str(method_spec["name"])
        kind = str(method_spec["kind"])
        if kind not in {"uniform", "markov", "checkpoint"}:
            raise ValueError(f"unknown benchmark method kind: {kind}")
        for motif_index, motif_row in enumerate(motifs):
            motif_id = str(motif_row["id"])
            motif = str(motif_row["sequence"]).upper()
            rng = random.Random(seed + method_index * 100000 + motif_index * 1000)
            length = max(len(motif) + 8, 24)
            if kind == "checkpoint":
                from rna_scaffold.generate import GenerationSettings, generate_candidates

                generation = dict(method_spec.get("generation", {}))
                generation.update(
                    num_candidates=budget,
                    seed=seed + method_index * 100000 + motif_index * 1000,
                )
                learned_candidates = generate_candidates(
                    method_spec["checkpoint"],
                    motif,
                    GenerationSettings(**generation),
                    device=method_spec.get("device", "cpu"),
                )
                generated = [
                    (candidate.full_sequence, candidate.motif_start)
                    for candidate in learned_candidates
                ]
            else:
                generated = []
                for _ in range(budget):
                    if kind == "uniform":
                        generated.append(_uniform(motif, length, rng))
                    else:
                        generated.append(_markov(motif, length, prior, rng))
            for candidate_index, (sequence, start) in enumerate(generated):
                metric = _metric(motif_id, motif, sequence, start)
                metric_rows[method].append(metric)
                candidate_rows.append(
                    {
                        "method": method,
                        "motif_id": motif_id,
                        "candidate_index": candidate_index,
                        "sequence": sequence,
                        "motif_start": start,
                        **asdict(metric),
                    }
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "candidates.csv"
    with candidates_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    motif_summaries = []
    for method in methods:
        for motif_row in motifs:
            rows = [row for row in metric_rows[method] if row.motif_id == motif_row["id"]]
            motif_summaries.append({"method": method, "motif_id": motif_row["id"], **asdict(summarize_candidates(rows))})
    motifs_path = output_dir / "motifs.csv"
    with motifs_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(motif_summaries[0]))
        writer.writeheader()
        writer.writerows(motif_summaries)
    summary = {
        "methods": [
            {"method": method, **asdict(summarize_candidates(metric_rows[method]))}
            for method in methods
        ],
        "candidate_budget_per_motif": budget,
        "seed": seed,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "seed": seed,
        "smoke_test": smoke_test,
        "python": sys.version,
        "platform": platform.platform(),
        "runtime_seconds": time.time() - started,
        "artifacts": {
            path.name: _sha256(path) for path in (candidates_path, motifs_path, summary_path)
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
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
