import torch

from train_3d import build_scheduler, mixed_precision_enabled, progress_enabled, select_training_device, wandb_enabled


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


def test_wandb_enabled_requires_explicit_false_to_disable():
    assert wandb_enabled({}) is True
    assert wandb_enabled({"enabled": True}) is True
    assert wandb_enabled({"enabled": False}) is False


def test_mixed_precision_enabled_only_for_cuda_when_requested():
    assert mixed_precision_enabled({"mixed_precision": True}, torch.device("cuda:0"))
    assert not mixed_precision_enabled({"mixed_precision": True}, torch.device("cpu"))
    assert not mixed_precision_enabled({"mixed_precision": False}, torch.device("cuda:0"))


def test_build_scheduler_warms_up_then_decays_learning_rate():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = build_scheduler(
        optimizer=optimizer,
        total_steps=10,
        warmup_steps=2,
        min_lr_ratio=0.1,
    )

    lrs = []
    for _ in range(5):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert lrs[0] < lrs[1]
    assert lrs[-1] < lrs[1]
