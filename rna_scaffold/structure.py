from __future__ import annotations

import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence

from rna_scaffold.generate import generate_rna_sequence
from rna_scaffold.utils import validate_rna_sequence


Runner = Callable[[Sequence[str], Path], tuple[int, str, str]]


class Rna3DStatus(str, Enum):
    PREDICTOR_NOT_CONFIGURED = "predictor_not_configured"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class Rna3DResult:
    motif: str
    sequence: str
    fasta_path: Path
    status: Rna3DStatus
    structure_path: Path | None
    message: str


def write_fasta(sequence: str, path: str | Path, name: str = "rna_candidate") -> Path:
    sequence = sequence.strip().upper().replace("T", "U")
    if not validate_rna_sequence(sequence):
        raise ValueError("sequence must contain only A, U, C, and G.")
    fasta_path = Path(path)
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    fasta_path.write_text(f">{name}\n{sequence}\n", encoding="utf-8")
    return fasta_path


def generate_rna_and_prepare_3d(
    motif: str,
    output_dir: str | Path = "outputs/rna_3d",
    name: str = "rna_candidate",
    predictor_command: Sequence[str] | None = None,
    num_candidates: int = 128,
    min_total_length: int | None = None,
    max_total_length: int | None = None,
    rng_seed: int | None = None,
    runner: Runner | None = None,
) -> Rna3DResult:
    motif = motif.strip().upper().replace("T", "U")
    sequence = generate_rna_sequence(
        motif=motif,
        num_candidates=num_candidates,
        min_total_length=min_total_length,
        max_total_length=max_total_length,
        rng_seed=rng_seed,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = write_fasta(sequence=sequence, path=out_dir / f"{name}.fa", name=name)
    structure_path = out_dir / f"{name}.pdb"

    if predictor_command is None:
        return Rna3DResult(
            motif=motif,
            sequence=sequence,
            fasta_path=fasta_path,
            status=Rna3DStatus.PREDICTOR_NOT_CONFIGURED,
            structure_path=None,
            message=(
                "RNA sequence and FASTA were generated. Configure a RhoFold/RhoFold+ "
                "command to predict the 3D structure."
            ),
        )

    command = _format_predictor_command(
        predictor_command,
        fasta_path=fasta_path,
        output_pdb=structure_path,
        output_dir=out_dir,
    )
    run = runner or _run_subprocess
    return_code, stdout, stderr = run(command, out_dir)
    if return_code != 0 or not structure_path.exists():
        detail = stderr.strip() or stdout.strip() or f"exit code {return_code}"
        return Rna3DResult(
            motif=motif,
            sequence=sequence,
            fasta_path=fasta_path,
            status=Rna3DStatus.FAILED,
            structure_path=None,
            message=f"3D predictor failed: {detail}",
        )

    return Rna3DResult(
        motif=motif,
        sequence=sequence,
        fasta_path=fasta_path,
        status=Rna3DStatus.COMPLETE,
        structure_path=structure_path,
        message=stdout.strip() or "3D structure prediction completed.",
    )


def _format_predictor_command(
    command: Sequence[str],
    fasta_path: Path,
    output_pdb: Path,
    output_dir: Path,
) -> list[str]:
    replacements = {
        "fasta": str(fasta_path),
        "output_pdb": str(output_pdb),
        "output_dir": str(output_dir),
    }
    return [part.format(**replacements) for part in command]


def _run_subprocess(command: Sequence[str], cwd: Path) -> tuple[int, str, str]:
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr
