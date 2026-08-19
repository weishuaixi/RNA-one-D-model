import json
import subprocess
import sys


def test_release_audit_records_truthful_check_states(tmp_path):
    output = tmp_path / "audit.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/verify_scaffold_release.py",
            "--output",
            str(output),
            "--skip-tests",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    audit = json.loads(output.read_text(encoding="utf-8"))
    assert audit["checks"]
    assert {check["status"] for check in audit["checks"]} <= {
        "passed",
        "failed",
        "not_run",
        "unavailable",
    }
    assert audit["server_training"]["status"] == "not_run"
