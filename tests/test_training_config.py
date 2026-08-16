from pathlib import Path

import yaml


def test_a800_config_is_accuracy_oriented_and_memory_safe():
    config_path = Path(__file__).parents[1] / "configs" / "train_scaffold_a800.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["data"]["max_target_length"] == 512
    assert config["model"]["d_model"] >= 768
    assert config["model"]["num_layers"] >= 12
    assert config["model"]["activation_checkpointing"] is True
    assert config["model"]["pretrained"]["kind"] == "rna_fm"
    assert config["model"]["pretrained"]["freeze"] is True
    assert config["trainer"]["args"]["precision"] == "bf16-mixed"
    assert config["trainer"]["args"]["accumulate_grad_batches"] >= 2
