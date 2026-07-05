import torch
import pytest

from rna_scaffold_3d.losses import masked_coordinate_mse, masked_pairwise_distance_mse
from rna_scaffold_3d.model import Rna3DCoordinatePredictor
from rna_scaffold_3d.pdb_writer import coordinates_to_pdb


def test_rna_3d_coordinate_predictor_returns_one_xyz_per_base():
    model = Rna3DCoordinatePredictor(d_model=32, nhead=4, num_layers=1, dim_feedforward=64)
    input_ids = torch.tensor([[1, 2, 3, 0]])
    padding_mask = torch.tensor([[False, False, False, True]])

    coords = model(input_ids=input_ids, padding_mask=padding_mask)

    assert coords.shape == (1, 4, 3)


def test_masked_coordinate_losses_ignore_invalid_positions():
    pred = torch.tensor([[[1.0, 1.0, 1.0], [100.0, 100.0, 100.0]]])
    target = torch.tensor([[[2.0, 1.0, 1.0], [0.0, 0.0, 0.0]]])
    mask = torch.tensor([[True, False]])

    assert masked_coordinate_mse(pred, target, mask).item() == pytest.approx(1.0 / 3.0)
    assert masked_pairwise_distance_mse(pred, target, mask).item() == 0.0


def test_coordinates_to_pdb_writes_pseudo_atom_per_residue():
    pdb = coordinates_to_pdb(
        sequence="AUG",
        coords=torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        ),
    )

    assert "ATOM" in pdb
    assert " C4'" in pdb
    assert "  A A   1" in pdb
    assert pdb.endswith("END\n")
