import torch

from train_3d import progress_enabled, select_training_device


def test_select_training_device_uses_configured_cuda_index_when_gpu_available():
    device = select_training_device(
        trainer_cfg={"accelerator": "gpu", "cuda_device": 1},
        cuda_available=True,
    )

    assert device == torch.device("cuda:1")


def test_select_training_device_falls_back_to_cpu_when_gpu_not_requested():
    device = select_training_device(
        trainer_cfg={"accelerator": "cpu", "cuda_device": 1},
        cuda_available=True,
    )

    assert device == torch.device("cpu")


def test_progress_enabled_defaults_to_true_and_can_be_disabled():
    assert progress_enabled({}) is True
    assert progress_enabled({"show_progress": False}) is False
