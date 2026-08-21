from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_REMOVED_RUNTIME_TERMS = (
    "position" + "_logits",
    "length" + "_logits",
    "position" + "_loss_weight",
    "length" + "_loss_weight",
    "train" + "_3d",
    "checkpoints_scaffold_a800" + "_mmseqs80",
)
_REMOVED_FALLBACK_PATTERN = re.compile(
    "(?i)(?:v" + "1.{0,40}fallback|fallback.{0,40}v" + "1)"
)


def _run(name: str, command: list[str]) -> dict:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    return {
        "name": name,
        "status": "passed" if completed.returncode == 0 else "failed",
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_files(root: Path) -> list[Path]:
    paths = sorted((root / "rna_scaffold").rglob("*.py"))
    paths.extend(sorted((root / "scripts").rglob("*.py")))
    paths.extend(sorted(root.glob("*.py")))
    paths.extend(sorted((root / "configs").rglob("*.yaml")))
    paths.extend(sorted((root / "configs").rglob("*.yml")))
    return paths


def _audit_v2_runtime_boundary(root: Path = ROOT) -> dict:
    violations = []
    for path in _runtime_files(root):
        text = path.read_text(encoding="utf-8")
        matched = [
            term
            for term in _REMOVED_RUNTIME_TERMS
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", text)
        ]
        if _REMOVED_FALLBACK_PATTERN.search(text):
            matched.append("legacy fallback wording")
        if matched:
            violations.append(
                {
                    "path": str(path.relative_to(root)),
                    "terms": matched,
                }
            )
    return {
        "name": "V2 runtime boundary",
        "status": "failed" if violations else "passed",
        "violations": violations,
    }


def run_cpu_tiny_v2_workflow(work_dir: str | Path) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from rna_scaffold.checkpoints import load_scaffold_checkpoint
    from rna_scaffold.data import RnaMotifDenoisingDataset
    from rna_scaffold.evaluation import CandidateMetric, summarize_candidates
    from rna_scaffold.generate import GenerationSettings, generate_candidates
    from rna_scaffold.lightning_module import RnaScaffoldLitModule
    from rna_scaffold.records import RnaSequenceRecord
    from rna_scaffold.tokenizer import RnaTokenizer

    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(17)
    tokenizer = RnaTokenizer()
    dataset = RnaMotifDenoisingDataset(
        records=[RnaSequenceRecord("tiny", "AACCGCGGUUAA", "tiny-cluster", "release-audit")],
        tokenizer=tokenizer,
        max_length=12,
        min_motif_length=4,
        max_motif_length=4,
        motif_length_buckets=None,
        min_flank_length=1,
        min_total_scaffold_length=2,
        preferred_total_scaffold_length=2,
        full_mask_probability=1.0,
        span_mask_probability=0.0,
        seed=17,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    hyper_parameters = {
        "vocab_size": tokenizer.vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
        "d_model": 16,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 32,
        "dropout": 0.0,
        "max_length": 12,
        "activation_checkpointing": False,
        "pretrained": {"kind": "none", "mode": "frozen"},
        "lr": 1e-3,
        "weight_decay": 0.0,
        "left_length_loss_weight": 0.25,
        "right_length_loss_weight": 0.25,
    }
    module = RnaScaffoldLitModule(**hyper_parameters)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
    module.train()
    optimizer.zero_grad(set_to_none=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"You are trying to `self\.log\(\)`.*",
            category=UserWarning,
        )
        training = module.training_step(batch, batch_idx=0)
    loss = training["loss"]
    loss.backward()
    optimizer.step()

    checkpoint_path = work_path / "tiny-v2.ckpt"
    checkpoint = {
        "state_dict": module.state_dict(),
        "hyper_parameters": hyper_parameters,
    }
    module.on_save_checkpoint(checkpoint)
    torch.save(checkpoint, checkpoint_path)
    loaded = load_scaffold_checkpoint(checkpoint_path, device="cpu")

    motif = "GCGG"
    generated = generate_candidates(
        checkpoint_path,
        motif,
        GenerationSettings(
            num_candidates=2,
            max_length=12,
            min_scaffold_length=2,
            min_flank_length=1,
            denoise_steps=2,
            top_p=1.0,
            max_attempts=32,
            seed=17,
        ),
        device="cpu",
    )
    rows = [
        CandidateMetric(
            motif_id="tiny-motif",
            sequence=candidate.full_sequence,
            valid=set(candidate.full_sequence) <= {"A", "U", "C", "G"},
            motif_preserved=candidate.motif_preserved,
            total_length=candidate.total_length,
            gc_fraction=candidate.gc_fraction,
            failure=None if candidate.motif_preserved else "motif_not_preserved",
            rnafold_status="not_run",
        )
        for candidate in generated
    ]
    summary = summarize_candidates(rows)
    return {
        "architecture_version": checkpoint["architecture_version"],
        "optimizer_steps": 1,
        "training_loss_is_finite": bool(torch.isfinite(loss.detach()).item()),
        "checkpoint_loaded": loaded.max_length == 12,
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "candidate_count": summary.count,
        "motif_preservation_rate": summary.motif_preservation_rate,
        "unique_rate": summary.unique_rate,
        "rnafold_status": "not_run",
    }


def _run_cpu_tiny_v2_check() -> dict:
    try:
        with tempfile.TemporaryDirectory(prefix="rna-scaffold-v2-audit-") as temporary:
            evidence = run_cpu_tiny_v2_workflow(temporary)
    except Exception as error:  # noqa: BLE001 - the audit must record every workflow failure
        return {
            "name": "CPU tiny V2 end-to-end",
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
    expected = (
        evidence["architecture_version"] == 2
        and evidence["optimizer_steps"] == 1
        and evidence["training_loss_is_finite"]
        and evidence["checkpoint_loaded"]
        and evidence["candidate_count"] >= 2
        and evidence["motif_preservation_rate"] == 1.0
        and evidence["unique_rate"] == 1.0
        and evidence["rnafold_status"] == "not_run"
    )
    return {
        "name": "CPU tiny V2 end-to-end",
        "status": "passed" if expected else "failed",
        "evidence": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a truthful local scaffold release audit.")
    parser.add_argument("--output", default="outputs/scaffold_release_audit.json")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    checks = []
    checks.append(_audit_v2_runtime_boundary())
    checks.append(_run_cpu_tiny_v2_check())
    if args.skip_tests:
        checks.append(
            {
                "name": "pytest",
                "status": "not_run",
                "command": [sys.executable, "-m", "pytest", "-q"],
                "reason": "disabled by --skip-tests",
            }
        )
    else:
        checks.append(_run("pytest", [sys.executable, "-m", "pytest", "-q"]))
    for script in ("generate_scaffold.py", "validate_scaffolds.py", "benchmark_scaffolds.py"):
        checks.append(_run(f"{script} help", [sys.executable, script, "--help"]))

    rnafold = shutil.which("RNAfold")
    if rnafold:
        checks.append(_run("RNAfold version", [rnafold, "--version"]))
    else:
        checks.append(
            {
                "name": "RNAfold version",
                "status": "unavailable",
                "command": ["RNAfold", "--version"],
                "reason": "RNAfold executable not installed locally",
            }
        )
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.splitlines()
    tracked_artifacts = [
        ROOT / "configs" / "train_scaffold_a800.yaml",
        ROOT / "configs" / "benchmark_scaffolds.yaml",
    ]
    audit = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "dirty_paths": dirty,
        "python": sys.version,
        "platform": platform.platform(),
        "checks": checks,
        "server_training": {
            "status": "not_run",
            "reason": "full A800 training and checkpoint metrics must be produced on the server",
        },
        "artifact_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in tracked_artifacts if path.is_file()
        },
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    if any(check["status"] == "failed" for check in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
