import json
import subprocess
import sys

from rna_scaffold.generate import ScaffoldCandidate, write_candidates_jsonl


def test_generate_cli_help():
    completed = subprocess.run(
        [sys.executable, "generate_scaffold.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--checkpoint" in completed.stdout
    assert "--motif" in completed.stdout


def test_candidate_jsonl_is_atomic_and_machine_readable(tmp_path):
    output = tmp_path / "candidates.jsonl"
    candidate = ScaffoldCandidate(
        candidate_id="candidate_0001",
        full_sequence="AAGCGGUU",
        left_sequence="AA",
        motif="GCGG",
        right_sequence="UU",
        motif_start=2,
        motif_end=6,
        total_length=8,
        normalized_log_probability=-0.2,
        checkpoint_sha256="abc",
        seed=42,
        gc_fraction=0.5,
        max_homopolymer=2,
        base_entropy=1.5,
        motif_preserved=True,
        valid=True,
        status="ok",
    )

    write_candidates_jsonl([candidate], output)

    assert not output.with_suffix(output.suffix + ".tmp").exists()
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["full_sequence"] == "AAGCGGUU"
