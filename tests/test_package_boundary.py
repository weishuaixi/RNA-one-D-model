from pathlib import Path


def test_local_3d_subsystem_is_not_packaged():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert 'include = ["rna_scaffold*"]' in pyproject
    assert "rna_scaffold_3d" not in pyproject
    assert "rna-train-3d" not in pyproject


def test_public_command_entry_points_are_declared():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert 'rna-generate-scaffold = "generate_scaffold:main"' in pyproject
    assert 'rna-validate-scaffolds = "validate_scaffolds:main"' in pyproject
    assert 'rna-benchmark-scaffolds = "benchmark_scaffolds:main"' in pyproject
