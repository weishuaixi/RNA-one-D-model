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
        "left_length_loss_weight": 0.25,
        "right_length_loss_weight": 0.25,
    }


def _write_tiny_checkpoint(path):
    hparams = _tiny_hparams()
    module = RnaScaffoldLitModule(**hparams)
    torch.save(
        {
            "architecture_version": 2,
            "state_dict": module.state_dict(),
            "hyper_parameters": hparams,
        },
        path,
    )


def _write_checkpoint_with_version(path, version):
    hparams = _tiny_hparams()
    module = RnaScaffoldLitModule(**hparams)
    payload = {"state_dict": module.state_dict(), "hyper_parameters": hparams}
    if version is not None:
        payload["architecture_version"] = version
    torch.save(payload, path)


def test_load_scaffold_checkpoint_reconstructs_exact_model(tmp_path):
    checkpoint = tmp_path / "tiny.ckpt"
    _write_tiny_checkpoint(checkpoint)

    loaded = load_scaffold_checkpoint(checkpoint)

    assert loaded.max_length == 32
    assert loaded.checkpoint_sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert loaded.model.training is False
    assert loaded.tokenizer.vocab_size == RnaTokenizer().vocab_size


def test_lightning_checkpoint_metadata_declares_v2_architecture():
    module = RnaScaffoldLitModule(**_tiny_hparams())
    checkpoint = {}

    module.on_save_checkpoint(checkpoint)

    assert checkpoint["architecture_version"] == 2


def test_load_scaffold_checkpoint_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scaffold_checkpoint(tmp_path / "missing.ckpt")


def test_load_scaffold_checkpoint_rejects_missing_hyperparameters(tmp_path):
    checkpoint = tmp_path / "invalid.ckpt"
    torch.save({"architecture_version": 2, "state_dict": {}}, checkpoint)

    with pytest.raises(CheckpointCompatibilityError, match="hyper_parameters"):
        load_scaffold_checkpoint(checkpoint)


@pytest.mark.parametrize("version", [None, 1], ids=["missing", "v1"])
def test_load_scaffold_checkpoint_rejects_non_v2_architecture(tmp_path, version):
    checkpoint = tmp_path / f"invalid-{version}.ckpt"
    _write_checkpoint_with_version(checkpoint, version)

    with pytest.raises(CheckpointCompatibilityError, match="architecture_version 2"):
        load_scaffold_checkpoint(checkpoint)
