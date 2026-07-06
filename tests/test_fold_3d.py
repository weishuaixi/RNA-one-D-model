from pathlib import Path

import torch

from fold_3d import fold_sequence_with_checkpoint, generate_sequence_from_motif, save_fold_outputs
from rna_scaffold_3d.rhofold import RhoFoldConfig, RhoFoldModel


def test_generate_sequence_from_motif_returns_complete_one_d_rna():
    sequence = generate_sequence_from_motif("GCGG", num_candidates=8, rng_seed=3)

    assert set(sequence).issubset({"A", "U", "C", "G"})
    assert "GCGG" in sequence


def test_fold_sequence_with_checkpoint_uses_local_rhofold_model(tmp_path: Path):
    config = {
        "d_model": 32,
        "pair_dim": 16,
        "msa_dim": 24,
        "nhead": 4,
        "num_e2e_layers": 1,
        "num_structure_layers": 1,
        "dim_feedforward": 64,
        "num_distance_bins": 8,
        "recycle_iters": 1,
    }
    model = RhoFoldModel(RhoFoldConfig(**config))
    checkpoint_path = tmp_path / "rhofold.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": {"model": config}}, checkpoint_path)

    result = fold_sequence_with_checkpoint("AUGC", checkpoint_path)

    assert result.sequence == "AUGC"
    assert result.coords.shape == (4, 27, 3)
    assert result.plddt.shape == (4,)


def test_save_fold_outputs_writes_tensor_file_and_complete_pdb(tmp_path: Path):
    config = {
        "d_model": 32,
        "pair_dim": 16,
        "msa_dim": 24,
        "nhead": 4,
        "num_e2e_layers": 1,
        "num_structure_layers": 1,
        "dim_feedforward": 64,
        "num_distance_bins": 8,
        "recycle_iters": 1,
    }
    model = RhoFoldModel(RhoFoldConfig(**config))
    checkpoint_path = tmp_path / "rhofold.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": {"model": config}}, checkpoint_path)
    result = fold_sequence_with_checkpoint("AUGC", checkpoint_path)

    tensor_path = tmp_path / "fold.pt"
    pdb_path = tmp_path / "fold.pdb"
    save_fold_outputs(result, tensor_path=tensor_path, pdb_path=pdb_path)

    assert tensor_path.exists()
    pdb_text = pdb_path.read_text(encoding="utf-8")
    assert pdb_text.endswith("END\n")
    assert sum(1 for line in pdb_text.splitlines() if line.startswith("ATOM")) == len(result.sequence) * 27
