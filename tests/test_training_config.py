from pathlib import Path

import yaml
from torch import nn

from rna_scaffold import lightning_module
from rna_scaffold.lightning_module import RnaScaffoldLitModule


class TinyConfiguredEncoder(nn.Module):
    output_dim = 6

    def forward(self, input_ids, attention_mask):
        raise AssertionError("configuration construction must not run a forward pass")


def test_a800_config_is_accuracy_oriented_and_memory_safe():
    config_path = Path(__file__).parents[1] / "configs" / "train_scaffold_a800.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["data"]["train_data"] == "data/processed/rna_merged_mmseqs80.csv"
    assert config["data"]["cluster_manifest"] == "outputs/mmseqs80_search/rna80_connected_components.tsv"
    assert "train_sequences.csv" not in config_path.read_text(encoding="utf-8")
    assert config["data"]["expected_partition_counts"] == {
        "train": 1513,
        "validation": 133,
        "test": 169,
    }
    assert config["data"]["cluster_thresholds"] == {
        "min_sequence_identity": 0.8,
        "min_bidirectional_coverage": 0.8,
    }
    assert config["data"]["max_target_length"] == 512
    assert config["data"]["min_motif_length"] == 4
    assert config["data"]["max_motif_length"] is None
    assert config["data"]["min_flank_length"] == 2
    assert config["data"]["min_total_scaffold_length"] == 8
    assert config["data"]["preferred_total_scaffold_length"] == 24
    assert config["data"]["motif_length_buckets"][-1] == [128, 512, 0.05]
    assert config["data"]["full_mask_probability"] > 0
    assert config["data"]["span_mask_probability"] > 0
    assert config["model"]["d_model"] >= 768
    assert config["model"]["num_layers"] >= 12
    assert config["model"]["activation_checkpointing"] is True
    assert config["model"]["pretrained"]["kind"] == "rna_fm"
    assert config["model"]["pretrained"]["checkpoint"] == (
        ".cache/torch/hub/checkpoints/RNA-FM_pretrained.pth"
    )
    assert config["model"]["pretrained"]["mode"] == "lora"
    assert config["model"]["pretrained"]["lora_rank"] == 8
    assert config["model"]["pretrained"]["lora_alpha"] == 16.0
    assert config["model"]["pretrained"]["lora_dropout"] == 0.05
    assert config["model"]["pretrained"]["lora_last_n_layers"] == 4
    assert config["model"]["left_length_loss_weight"] == 0.25
    assert config["model"]["right_length_loss_weight"] == 0.25
    assert "length_loss_weight" not in config["model"]
    assert "position_loss_weight" not in config["model"]
    assert config["model"]["warmup_fraction"] == 0.05
    assert config["model"]["min_lr_fraction"] == 0.02
    assert config["model"]["label_smoothing"] > 0
    assert config["trainer"]["early_stopping_patience"] > 0
    assert config["trainer"]["args"]["precision"] == "bf16-mixed"
    assert config["trainer"]["args"]["accumulate_grad_batches"] >= 2


def test_a800_model_config_constructs_lightning_boundary(monkeypatch):
    config_path = Path(__file__).parents[1] / "configs" / "train_scaffold_a800.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        lightning_module,
        "build_pretrained_encoder",
        lambda _: TinyConfiguredEncoder(),
    )

    module = RnaScaffoldLitModule(vocab_size=12, pad_token_id=0, **config["model"])

    assert module.left_length_loss_weight == 0.25
    assert module.right_length_loss_weight == 0.25
