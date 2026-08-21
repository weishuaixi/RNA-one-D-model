import re
from pathlib import Path

FORBIDDEN_V1_RUNTIME_TERMS = (
    "position_logits",
    "length_logits",
    "position_loss_weight",
    "length_loss_weight",
    "train_3d",
    "checkpoints_scaffold_a800_mmseqs80",
)


def _runtime_files(root: Path) -> list[Path]:
    package_files = sorted((root / "rna_scaffold").rglob("*.py"))
    script_files = sorted((root / "scripts").rglob("*.py"))
    entry_points = sorted(root.glob("*.py"))
    configuration_files = sorted((root / "configs").rglob("*.yaml"))
    configuration_files.extend(sorted((root / "configs").rglob("*.yml")))
    return package_files + script_files + entry_points + configuration_files


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


def test_runtime_sources_and_configuration_do_not_reference_v1_boundaries():
    root = Path(__file__).resolve().parents[1]
    violations: dict[str, list[str]] = {}

    for path in _runtime_files(root):
        text = path.read_text(encoding="utf-8")
        matched = [
            term
            for term in FORBIDDEN_V1_RUNTIME_TERMS
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", text)
        ]
        if re.search(r"(?i)(?:v1.{0,40}fallback|fallback.{0,40}v1)", text):
            matched.append("V1 fallback wording")
        if matched:
            violations[str(path.relative_to(root))] = matched

    assert violations == {}
