from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a truthful local scaffold release audit.")
    parser.add_argument("--output", default="outputs/scaffold_release_audit.json")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    checks = []
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
