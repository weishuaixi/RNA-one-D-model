from pathlib import Path

import torch

from fold_3d import (
    _iterative_decode_sequence,
    fold_motif_with_checkpoint,
    fold_sequence_with_checkpoint,
    generate_sequence_from_motif,
    save_fold_outputs,
)
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


def test_fold_motif_with_joint_checkpoint_generates_sequence_then_structure(tmp_path: Path):
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
    checkpoint_path = tmp_path / "joint_rhofold.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": {"model": config}}, checkpoint_path)

    result = fold_motif_with_checkpoint(
        "GCGG",
        checkpoint_path,
        num_candidates=2,
        min_total_length=12,
        max_total_length=12,
        rng_seed=3,
    )

    assert len(result.sequence) == 12
    assert "GCGG" in result.sequence
    assert result.coords.shape == (12, 27, 3)


def test_iterative_decode_preserves_known_motif_and_resolves_all_masks():
    config = RhoFoldConfig(
        d_model=32,
        pair_dim=16,
        msa_dim=24,
        nhead=4,
        num_e2e_layers=1,
        num_structure_layers=1,
        dim_feedforward=64,
        num_distance_bins=8,
    )
    model = RhoFoldModel(config).eval()
    input_ids = torch.tensor([[5, 5, 3, 4, 5, 5]])
    padding_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    with torch.no_grad():
        decoded = _iterative_decode_sequence(model, input_ids, padding_mask, denoise_steps=3)

    assert decoded[0, 2:4].tolist() == [3, 4]
    assert not decoded.eq(5).any()
    assert set(decoded.flatten().tolist()) <= {1, 2, 3, 4}


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
