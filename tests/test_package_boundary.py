from pathlib import Path


def test_local_3d_subsystem_is_not_shipped():
    root = Path(__file__).resolve().parents[1]
    forbidden = [
        root / "rna_scaffold_3d",
        root / "train_3d.py",
        root / "fold_3d.py",
        root / "evaluate_3d.py",
    ]

    assert not [str(path) for path in forbidden if path.exists()]
