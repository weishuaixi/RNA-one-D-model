from pathlib import Path

from rna_scaffold.structure import (
    Rna3DStatus,
    generate_rna_and_prepare_3d,
    write_fasta,
)


def test_write_fasta_writes_generated_rna_sequence(tmp_path: Path):
    fasta_path = tmp_path / "candidate.fa"

    write_fasta(sequence="AUGCGU", path=fasta_path, name="candidate_1")

    assert fasta_path.read_text(encoding="utf-8") == ">candidate_1\nAUGCGU\n"


def test_generate_rna_and_prepare_3d_returns_sequence_and_fasta_without_predictor(tmp_path: Path):
    result = generate_rna_and_prepare_3d(
        motif="GCGG",
        output_dir=tmp_path,
        num_candidates=8,
        rng_seed=5,
    )

    assert set(result.sequence).issubset({"A", "U", "C", "G"})
    assert "GCGG" in result.sequence
    assert result.fasta_path.exists()
    assert result.status == Rna3DStatus.PREDICTOR_NOT_CONFIGURED
    assert result.structure_path is None
    assert "RhoFold" in result.message


def test_generate_rna_and_prepare_3d_invokes_configured_predictor(tmp_path: Path):
    calls = []

    def fake_runner(command, cwd):
        calls.append((command, cwd))
        structure_path = Path(command[-1])
        structure_path.write_text("HEADER test structure\n", encoding="utf-8")
        return 0, "ok", ""

    result = generate_rna_and_prepare_3d(
        motif="GCGG",
        output_dir=tmp_path,
        predictor_command=["fake-rhofold", "{fasta}", "{output_pdb}"],
        runner=fake_runner,
        num_candidates=8,
        rng_seed=5,
    )

    assert result.status == Rna3DStatus.COMPLETE
    assert result.structure_path is not None
    assert result.structure_path.exists()
    assert calls
    assert str(result.fasta_path) in calls[0][0]
