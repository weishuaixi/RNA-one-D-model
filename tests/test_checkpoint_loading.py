import hashlib

import pytest
import torch

from rna_scaffold.checkpoints import CheckpointCompatibilityError, load_scaffold_checkpoint
from rna_scaffold.lightning_module import RnaScaffoldLitModule
from rna_scaffold.tokenizer import RnaTokenizer


def _tiny_hparams() -> dict:
    tokenizer = RnaTokenizer()
    return {
        "vocab_size": tokenizer.vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
        "d_model": 16,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 32,
        "dropout": 0.0,
        "max_length": 32,
        "activation_checkpointing": False,
        "pretrained": {"kind": "none"},
        "lr": 1e-3,
        "weight_decay": 0.0,
        "length_loss_weight": 0.25,
        "position_loss_weight": 0.25,
    }


def _write_tiny_checkpoint(path):
    hparams = _tiny_hparams()
    module = RnaScaffoldLitModule(**hparams)
    torch.save({"state_dict": module.state_dict(), "hyper_parameters": hparams}, path)


def test_load_scaffold_checkpoint_reconstructs_exact_model(tmp_path):
    checkpoint = tmp_path / "tiny.ckpt"
    _write_tiny_checkpoint(checkpoint)

    loaded = load_scaffold_checkpoint(checkpoint)

    assert loaded.max_length == 32
    assert loaded.checkpoint_sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert loaded.model.training is False
    assert loaded.tokenizer.vocab_size == RnaTokenizer().vocab_size


def test_load_scaffold_checkpoint_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scaffold_checkpoint(tmp_path / "missing.ckpt")


def test_load_scaffold_checkpoint_rejects_missing_hyperparameters(tmp_path):
    checkpoint = tmp_path / "invalid.ckpt"
    torch.save({"state_dict": {}}, checkpoint)

    with pytest.raises(CheckpointCompatibilityError, match="hyper_parameters"):
        load_scaffold_checkpoint(checkpoint)
