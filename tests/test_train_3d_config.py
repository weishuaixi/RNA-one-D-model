import json
import random
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import Dataset

from train_3d import (
    CroppedRnaDataset,
    build_scheduler,
    checkpoint_selection,
    dataset_membership,
    mask_sequence_inputs,
    mixed_precision_enabled,
    normalize_accumulated_gradients,
    progress_enabled,
    select_training_device,
    sequence_reconstruction_loss,
    restore_training_checkpoint,
    save_training_checkpoint,
    training_artifact_provenance,
    training_semantics_fingerprint,
    validate_config,
    wandb_enabled,
)
from rna_scaffold_3d.rhofold import RhoFoldConfig, RhoFoldModel
from rna_scaffold_3d.data import StanfordRna3DRecord, collate_3d_batch
from rna_scaffold_3d.splitting import (
    exclude_external_holdout,
    leakage_safe_train_val_split,
)


class _RecordDataset(Dataset):
    def __init__(self, target_sequences):
        self.records = [
            StanfordRna3DRecord(
                target_id=target_id,
                sequence=sequence,
                coords=torch.zeros((len(sequence), 1, 3)),
                coord_mask=torch.ones((len(sequence), 1), dtype=torch.bool),
            )
            for target_id, sequence in target_sequences
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class _ItemDataset(Dataset):
    def __init__(self, lengths):
        self.items = []
        for index, length in enumerate(lengths):
            self.items.append({
                "target_id": f"R{index}",
                "sequence": "A" * length,
                "input_ids": torch.ones(length, dtype=torch.long),
                "coords": torch.arange(length, dtype=torch.float32)
                .view(length, 1)
                .expand(-1, 3),
                "coord_mask": torch.ones(length, dtype=torch.bool),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def test_validation_multicrop_covers_start_center_end_and_weights_each_rna_once():
    dataset = CroppedRnaDataset(
        _ItemDataset([10, 3]),
        crop_length=4,
        random_crop=False,
        deterministic_crops=3,
    )

    assert len(dataset) == 4
    long_windows = [dataset[index] for index in range(3)]
    assert [item["target_id"] for item in long_windows] == [
        "R0:0-4",
        "R0:3-7",
        "R0:6-10",
    ]
    assert sum(item["example_weight"] for item in long_windows) == pytest.approx(1.0)
    assert dataset[3]["target_id"] == "R1"
    assert dataset[3]["example_weight"] == 1.0
    batch = collate_3d_batch([dataset[0], dataset[3]])
    assert torch.allclose(
        batch["example_weights"], torch.tensor([1.0 / 3.0, 1.0])
    )


def test_validation_multicrop_auto_expands_to_cover_every_residue():
    dataset = CroppedRnaDataset(
        _ItemDataset([20]),
        crop_length=4,
        random_crop=False,
        deterministic_crops=3,
    )

    windows = [dataset[index] for index in range(len(dataset))]
    covered = set()
    for item in windows:
        start, stop = (
            int(value)
            for value in item["target_id"].rsplit(":", 1)[1].split("-")
        )
        covered.update(range(start, stop))

    assert len(windows) == 5
    assert covered == set(range(20))
    assert sum(item["example_weight"] for item in windows) == pytest.approx(1.0)


def test_checkpoint_selection_rejects_unknown_or_empty_metrics():
    with pytest.raises(ValueError, match="Unknown trainer.checkpoint_metric"):
        checkpoint_selection({"loss": 1.0}, "c1_lddt", "max", float("-inf"))

    value, valid, improved = checkpoint_selection(
        {"c1_lddt": 0.0, "c1_lddt_count": 0.0},
        "c1_lddt",
        "max",
        float("-inf"),
    )
    assert value == 0.0
    assert not valid
    assert not improved


def test_checkpoint_selection_requires_finite_value_and_tracks_mode():
    _, valid_nan, improved_nan = checkpoint_selection(
        {"loss": float("nan")}, "loss", "min", float("inf")
    )
    value, valid, improved = checkpoint_selection(
        {"c1_lddt": 60.0, "c1_lddt_count": 4.0},
        "c1_lddt",
        "max",
        55.0,
    )

    assert not valid_nan and not improved_nan
    assert value == 60.0
    assert valid and improved


def test_training_semantics_ignore_runtime_paths_but_detect_training_drift():
    base = {
        "seed": 7,
        "data": {
            "source": "cif_all_atom",
            "sequences_csv": "/old/train.csv",
            "cif_dir": "/old/cif",
            "cache_path": "/old/cache.pt",
            "crop_length": 128,
        },
        "model": {"d_model": 32, "dropout": 0.1},
        "optimizer": {"lr": 1e-3, "fape_weight": 1.0},
        "trainer": {
            "accelerator": "gpu",
            "cuda_device": 0,
            "checkpoint_dir": "/old/output",
            "batch_size": 2,
            "max_epochs": 10,
        },
    }
    moved = json.loads(json.dumps(base))
    moved["data"]["sequences_csv"] = "/new/train.csv"
    moved["data"]["cif_dir"] = "/new/cif"
    moved["trainer"]["cuda_device"] = 1
    moved["trainer"]["checkpoint_dir"] = "/new/output"
    members = [("R1", "AUGC")]

    reference = training_semantics_fingerprint(base, members, members, 20)

    assert training_semantics_fingerprint(
        moved, members, members, 20
    ) == reference
    changed_loss = json.loads(json.dumps(base))
    changed_loss["optimizer"]["fape_weight"] = 0.5
    assert training_semantics_fingerprint(
        changed_loss, members, members, 20
    ) != reference
    assert training_semantics_fingerprint(
        base, [("R2", "AUGC")], members, 20
    ) != reference
    assert training_semantics_fingerprint(
        base, members, members, 21
    ) != reference


def test_dataset_membership_detects_coordinate_label_changes():
    first = _RecordDataset([("R1", "AUGC")])
    second = _RecordDataset([("R1", "AUGC")])
    second.records[0].coords[0, 0, 0] = 0.25

    first_membership = dataset_membership(first)
    second_membership = dataset_membership(second)

    assert first_membership[0][:2] == second_membership[0][:2]
    assert first_membership[0][2] != second_membership[0][2]


def test_select_training_device_uses_configured_cuda_index_when_gpu_available():
    device = select_training_device(
        trainer_cfg={"accelerator": "gpu", "cuda_device": 1},
        cuda_available=True,
        cuda_device_count=2,
    )

    assert device == torch.device("cuda:1")


def test_select_training_device_rejects_missing_or_wrong_cuda_device():
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        select_training_device(
            {"accelerator": "gpu", "cuda_device": 0},
            cuda_available=False,
            cuda_device_count=0,
        )
    with pytest.raises(RuntimeError, match="detected 1 CUDA device"):
        select_training_device(
            {"accelerator": "gpu", "cuda_device": 1},
            cuda_available=True,
            cuda_device_count=1,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("accelerator", "tpu", "accelerator"),
        ("cuda_device", -1, "cuda_device"),
        ("num_workers", -1, "num_workers"),
        ("max_epochs", 0, "max_epochs"),
        ("accumulate_grad_batches", 0, "accumulate_grad_batches"),
        ("validation_crops", 0, "validation_crops"),
        ("gradient_clip_norm", 0.0, "gradient_clip_norm"),
        ("random_translation_scale", -1.0, "random_translation_scale"),
    ],
)
def test_validate_config_rejects_invalid_training_runtime_values(
    field, value, message
):
    cfg = {
        "data": {},
        "model": {
            "d_model": 16,
            "nhead": 4,
            "pair_dim": 8,
            "pair_heads": 2,
        },
        "optimizer": {},
        "trainer": {
            "accelerator": "cpu",
            "cuda_device": 0,
            "num_workers": 0,
            "batch_size": 1,
            "max_epochs": 1,
            "accumulate_grad_batches": 1,
        },
    }
    cfg["trainer"][field] = value

    with pytest.raises(ValueError, match=message):
        validate_config(cfg)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("lr", 0.0, "optimizer.lr"),
        ("min_lr", -1e-6, "optimizer.min_lr"),
        ("weight_decay", -0.1, "weight_decay"),
        ("warmup_steps", -1, "warmup_steps"),
        ("fape_weight", 0.0, "fape_weight"),
        ("fape_weight", float("nan"), "fape_weight"),
        ("clash_weight", -0.1, "clash_weight"),
        ("clash_weight", float("inf"), "clash_weight"),
        ("torsion_weight", -0.1, "torsion_weight"),
        ("base_orientation_weight", -0.1, "base_orientation_weight"),
    ],
)
def test_validate_config_rejects_disabled_or_negative_loss_semantics(
    field, value, message
):
    cfg = {
        "data": {},
        "model": {
            "d_model": 16,
            "nhead": 4,
            "pair_dim": 8,
            "pair_heads": 2,
        },
        "optimizer": {"lr": 1e-3},
        "trainer": {
            "accelerator": "cpu",
            "cuda_device": 0,
            "num_workers": 0,
            "batch_size": 1,
            "max_epochs": 1,
            "accumulate_grad_batches": 1,
        },
    }
    cfg["optimizer"][field] = value

    with pytest.raises(ValueError, match=message):
        validate_config(cfg)


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


def test_gradient_accumulation_weights_uneven_microbatches_by_examples():
    full_model = torch.nn.Linear(1, 1, bias=False)
    accumulated_model = torch.nn.Linear(1, 1, bias=False)
    accumulated_model.load_state_dict(full_model.state_dict())
    inputs = torch.tensor([[1.0], [2.0], [4.0]])
    targets = torch.tensor([[0.0], [1.0], [3.0]])

    full_loss = (full_model(inputs) - targets).square().mean()
    full_loss.backward()

    for indices in (slice(0, 2), slice(2, 3)):
        batch_loss = (
            accumulated_model(inputs[indices]) - targets[indices]
        ).square().mean()
        batch_size = inputs[indices].size(0)
        (batch_loss * batch_size).backward()
    normalize_accumulated_gradients(
        accumulated_model, example_count=len(inputs)
    )

    assert torch.allclose(
        accumulated_model.weight.grad,
        full_model.weight.grad,
        atol=1e-7,
    )


def test_mask_sequence_inputs_can_keep_a_motif_and_mask_the_scaffold():
    torch.manual_seed(4)
    input_ids = torch.tensor([[1, 2, 3, 4, 1, 2]])
    padding_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    masked, selected = mask_sequence_inputs(
        input_ids,
        padding_mask,
        mask_token_id=5,
        mask_probability=0.0,
        scaffold_mask_probability=1.0,
        motif_length=2,
        training=True,
    )

    assert selected.sum().item() == 4
    assert torch.all(masked[selected] == 5)
    assert torch.equal(masked[~selected], input_ids[~selected])


def test_single_residue_sequence_still_has_reconstruction_supervision():
    input_ids = torch.tensor([[0, 3, 0]])
    padding_mask = torch.tensor([[True, False, True]])

    masked, selected = mask_sequence_inputs(
        input_ids,
        padding_mask,
        mask_token_id=5,
        mask_probability=0.0,
        scaffold_mask_probability=0.0,
        motif_length=1,
        training=True,
    )

    assert torch.equal(selected, torch.tensor([[False, True, False]]))
    assert masked[0, 1].item() == 5


def test_sequence_reconstruction_loss_backpropagates_inside_joint_objective():
    logits = torch.zeros((1, 3, 6), requires_grad=True)
    targets = torch.tensor([[1, 2, 3]])
    selected = torch.tensor([[True, False, True]])

    loss = sequence_reconstruction_loss(logits, targets, selected)
    loss.backward()

    assert loss.item() > 0
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0


def test_sequence_reconstruction_loss_weights_rnas_not_masked_token_counts():
    torch.manual_seed(91)
    logits = torch.randn(2, 8, 6)
    targets = torch.randint(0, 6, (2, 8))
    selected = torch.zeros(2, 8, dtype=torch.bool)
    selected[0, :2] = True
    selected[1, :8] = True

    batch_loss = sequence_reconstruction_loss(logits, targets, selected)
    individual_losses = torch.stack(
        [
            sequence_reconstruction_loss(
                logits[index:index + 1],
                targets[index:index + 1],
                selected[index:index + 1],
            )
            for index in range(2)
        ]
    )

    assert torch.allclose(batch_loss, individual_losses.mean(), atol=1e-7)


def test_training_checkpoint_restores_model_optimizer_scheduler_and_rng(tmp_path: Path):
    import numpy as np

    config = RhoFoldConfig(
        d_model=16,
        pair_dim=8,
        msa_dim=8,
        nhead=4,
        pair_heads=2,
        num_e2e_layers=1,
        num_structure_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    model = RhoFoldModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    loss = model(torch.tensor([[1, 2, 3, 4]])).pow(2).mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    saved_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    random.seed(91)
    torch.manual_seed(92)
    np.random.seed(95)
    data_generators = {
        "train": torch.Generator().manual_seed(93),
        "val": torch.Generator().manual_seed(94),
    }
    torch.rand(3, generator=data_generators["train"])
    torch.rand(2, generator=data_generators["val"])
    checkpoint = tmp_path / "resume.pt"
    artifact_provenance = {
        "split_manifest_sha256": "split-sha",
        "holdout_manifest_sha256": "holdout-sha",
    }
    save_training_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        {"model": {}},
        epoch=3,
        val_loss=1.2,
        best_val=42.0,
        data_generators=data_generators,
        training_semantics="test-semantics",
        training_provenance=artifact_provenance,
    )
    saved_checkpoint = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    assert saved_checkpoint["format_version"] == 5
    assert saved_checkpoint["training_provenance"] == artifact_provenance
    with pytest.raises(ValueError, match="different checkpoint metric"):
        restore_training_checkpoint(
            checkpoint,
            model,
            optimizer,
            scheduler,
            torch.device("cpu"),
            expected_checkpoint_metric="loss",
        )
    with pytest.raises(ValueError, match="different checkpoint mode"):
        restore_training_checkpoint(
            checkpoint,
            model,
            optimizer,
            scheduler,
            torch.device("cpu"),
            expected_checkpoint_mode="min",
        )
    with pytest.raises(ValueError, match="semantics"):
        restore_training_checkpoint(
            checkpoint,
            model,
            optimizer,
            scheduler,
            torch.device("cpu"),
            expected_training_semantics="different",
        )
    with pytest.raises(ValueError, match="artifact provenance"):
        restore_training_checkpoint(
            checkpoint,
            model,
            optimizer,
            scheduler,
            torch.device("cpu"),
            expected_training_provenance={
                **artifact_provenance,
                "split_manifest_sha256": "different",
            },
        )
    expected_python = random.random()
    expected_torch = torch.rand(1)
    expected_numpy = np.random.random(3)
    expected_train_data = torch.rand(4, generator=data_generators["train"])
    expected_val_data = torch.rand(4, generator=data_generators["val"])
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)
    random.seed(1)
    torch.manual_seed(1)
    np.random.seed(1)
    data_generators["train"].manual_seed(1)
    data_generators["val"].manual_seed(1)

    epoch, best = restore_training_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        torch.device("cpu"),
        data_generators=data_generators,
        expected_checkpoint_metric="c1_lddt",
        expected_checkpoint_mode="max",
        expected_training_semantics="test-semantics",
        expected_training_provenance=artifact_provenance,
    )

    assert epoch == 3
    assert best == 42.0
    assert all(torch.equal(model.state_dict()[name], value) for name, value in saved_state.items())
    assert random.random() == expected_python
    assert torch.equal(torch.rand(1), expected_torch)
    assert np.array_equal(np.random.random(3), expected_numpy)
    assert torch.equal(
        torch.rand(4, generator=data_generators["train"]),
        expected_train_data,
    )
    assert torch.equal(
        torch.rand(4, generator=data_generators["val"]),
        expected_val_data,
    )


def test_training_artifact_provenance_hashes_exact_manifests(tmp_path):
    split = tmp_path / "split.json"
    holdout = tmp_path / "holdout.json"
    split.write_text('{"split":1}', encoding="utf-8")
    holdout.write_text('{"holdout":1}', encoding="utf-8")
    provenance = training_artifact_provenance(
        {
            "data": {
                "external_holdout": {"manifest_path": str(holdout)}
            },
            "trainer": {
                "sequence_split": {"manifest_path": str(split)}
            },
        }
    )

    assert provenance["split_manifest_sha256"] != provenance[
        "holdout_manifest_sha256"
    ]
    split.write_text('{"split":2}', encoding="utf-8")
    assert training_artifact_provenance(
        {
            "data": {
                "external_holdout": {"manifest_path": str(holdout)}
            },
            "trainer": {
                "sequence_split": {"manifest_path": str(split)}
            },
        }
    )["split_manifest_sha256"] != provenance["split_manifest_sha256"]


def test_resumed_stochastic_training_matches_uninterrupted_next_step(tmp_path: Path):
    config = RhoFoldConfig(
        d_model=16,
        pair_dim=8,
        msa_dim=8,
        nhead=4,
        pair_heads=2,
        num_e2e_layers=1,
        num_structure_layers=1,
        dim_feedforward=32,
        dropout=0.1,
        recycle_iters=2,
        random_recycle_training=True,
        equivariant_layers=0,
    )

    def make_training_state():
        current_model = RhoFoldModel(config).train()
        current_optimizer = torch.optim.AdamW(current_model.parameters(), lr=1e-3)
        current_scheduler = torch.optim.lr_scheduler.LambdaLR(
            current_optimizer, lambda step: 1.0
        )
        return current_model, current_optimizer, current_scheduler

    def stochastic_step(current_model, current_optimizer, current_scheduler, generator):
        input_ids = torch.randint(1, 5, (1, 5), generator=generator)
        current_optimizer.zero_grad(set_to_none=True)
        loss = current_model(input_ids).square().mean()
        loss.backward()
        current_optimizer.step()
        current_scheduler.step()

    random.seed(120)
    torch.manual_seed(121)
    model, optimizer, scheduler = make_training_state()
    train_generator = torch.Generator().manual_seed(122)
    stochastic_step(model, optimizer, scheduler, train_generator)
    checkpoint = tmp_path / "exact_resume.pt"
    save_training_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        {"model": {}},
        epoch=1,
        val_loss=1.0,
        best_val=1.0,
        data_generators={"train": train_generator},
    )

    stochastic_step(model, optimizer, scheduler, train_generator)
    uninterrupted = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    random.seed(999)
    torch.manual_seed(999)
    resumed_model, resumed_optimizer, resumed_scheduler = make_training_state()
    resumed_generator = torch.Generator().manual_seed(999)
    restore_training_checkpoint(
        checkpoint,
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        torch.device("cpu"),
        data_generators={"train": resumed_generator},
    )
    stochastic_step(
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        resumed_generator,
    )

    assert all(
        torch.equal(resumed_model.state_dict()[name], expected)
        for name, expected in uninterrupted.items()
    )


def test_leakage_safe_split_keeps_pdb_groups_and_near_duplicates_together(tmp_path: Path):
    shared = "ACGU" * 25
    near_duplicate = shared[:50] + "A" + shared[51:]
    dataset = _RecordDataset([
        ("1ABC_A", shared),
        ("2DEF_A", near_duplicate),
        ("3GHI_A", "A" * 100),
        ("3GHI_B", "C" * 100),
        ("4JKL_A", "G" * 100),
        ("5MNO_A", "U" * 100),
    ])

    train, val, audit = leakage_safe_train_val_split(
        dataset,
        val_fraction=0.34,
        seed=7,
        manifest_path=tmp_path / "split.json",
        kmer_size=8,
        jaccard_threshold=0.8,
    )
    train_indices, val_indices = set(train.indices), set(val.indices)

    assert (0 in train_indices) == (1 in train_indices)
    assert (2 in train_indices) == (3 in train_indices)
    assert not train_indices & val_indices
    assert audit.exact_sequence_overlap == 0
    assert audit.max_cross_split_jaccard < 0.8


def test_internal_split_does_not_depend_on_probabilistic_lsh(tmp_path: Path):
    shared = "ACGU" * 25
    near_duplicate = shared[:50] + "A" + shared[51:]
    dataset = _RecordDataset([
        ("1ABC_A", shared),
        ("2DEF_A", near_duplicate),
        ("3GHI_A", "A" * 100),
        ("4JKL_A", "G" * 100),
    ])

    with (
        patch(
            "rna_scaffold_3d.splitting._candidate_pairs",
            side_effect=AssertionError("internal split must be exhaustive"),
        ),
        patch(
            "rna_scaffold_3d.splitting._anchor_candidate_pairs",
            side_effect=AssertionError("internal split must be exhaustive"),
        ),
    ):
        train, _, audit = leakage_safe_train_val_split(
            dataset,
            0.5,
            7,
            manifest_path=tmp_path / "split.json",
            kmer_size=8,
            jaccard_threshold=0.8,
        )

    payload = json.loads((tmp_path / "split.json").read_text("utf-8"))
    assert (0 in train.indices) == (1 in train.indices)
    assert audit.cross_split_audit_exhaustive is True
    assert payload["format_version"] == 3
    assert payload["candidate_strategy"] == "exhaustive_length_bounded"


def test_split_manifest_is_reused_and_invalidated_by_dataset_change(tmp_path: Path):
    manifest = tmp_path / "split.json"
    dataset = _RecordDataset([
        ("A_A", "ACGU" * 10),
        ("B_A", "A" * 40),
        ("C_A", "C" * 40),
        ("D_A", "G" * 40),
    ])
    first_train, first_val, _ = leakage_safe_train_val_split(
        dataset, 0.25, 13, manifest_path=manifest, kmer_size=4
    )
    first_payload = json.loads(manifest.read_text(encoding="utf-8"))
    second_train, second_val, _ = leakage_safe_train_val_split(
        dataset, 0.25, 13, manifest_path=manifest, kmer_size=4
    )

    assert first_train.indices == second_train.indices
    assert first_val.indices == second_val.indices

    changed = _RecordDataset([
        ("A_A", "ACGU" * 10),
        ("B_A", "U" * 40),
        ("C_A", "C" * 40),
        ("D_A", "G" * 40),
    ])
    leakage_safe_train_val_split(
        changed, 0.25, 13, manifest_path=manifest, kmer_size=4
    )
    second_payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert first_payload["dataset_fingerprint"] != second_payload["dataset_fingerprint"]


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("train_indices", [0, 0, 2], "target IDs"),
        ("train_target_ids", ["WRONG", "C_A", "D_A"], "target IDs"),
    ],
)
def test_matching_split_manifest_fails_closed_when_corrupted(
    tmp_path,
    field,
    replacement,
    message,
):
    manifest = tmp_path / "split.json"
    dataset = _RecordDataset([
        ("A_A", "ACGU" * 10),
        ("B_A", "A" * 40),
        ("C_A", "C" * 40),
        ("D_A", "G" * 40),
    ])
    leakage_safe_train_val_split(
        dataset, 0.25, 13, manifest_path=manifest, kmer_size=4
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload[field] = replacement
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        leakage_safe_train_val_split(
            dataset, 0.25, 13, manifest_path=manifest, kmer_size=4
        )


def test_matching_split_manifest_rechecks_near_duplicate_boundary(
    tmp_path,
):
    shared = "ACGU" * 25
    near_duplicate = shared[:50] + "A" + shared[51:]
    dataset = _RecordDataset([
        ("1ABC_A", shared),
        ("2DEF_A", near_duplicate),
        ("3GHI_A", "A" * 100),
        ("4JKL_A", "G" * 100),
        ("5MNO_A", "U" * 100),
    ])
    manifest = tmp_path / "split.json"
    leakage_safe_train_val_split(
        dataset,
        0.4,
        7,
        manifest_path=manifest,
        kmer_size=8,
        jaccard_threshold=0.8,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    source_name = (
        "train" if 1 in payload["train_indices"] else "val"
    )
    destination_name = "val" if source_name == "train" else "train"
    source = payload[f"{source_name}_indices"]
    destination = payload[f"{destination_name}_indices"]
    destination_position = next(
        position
        for position, index in enumerate(destination)
        if index != 0
    )
    source_position = source.index(1)
    source[source_position], destination[destination_position] = (
        destination[destination_position],
        source[source_position],
    )
    for name in ("train", "val"):
        payload[f"{name}_target_ids"] = [
            dataset.records[index].target_id
            for index in payload[f"{name}_indices"]
        ]
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="near-duplicate"):
        leakage_safe_train_val_split(
            dataset,
            0.4,
            7,
            manifest_path=manifest,
            kmer_size=8,
            jaccard_threshold=0.8,
        )


def test_external_holdout_excludes_exact_and_near_duplicate_sequences(tmp_path: Path):
    random_source = random.Random(19)
    shared = "".join(random_source.choice("ACGU") for _ in range(120))
    near_duplicate = shared[:60] + ("A" if shared[60] != "A" else "C") + shared[61:]
    dataset = _RecordDataset([
        ("TRAIN_EXACT_A", shared),
        ("TRAIN_NEAR_A", near_duplicate),
        ("TRAIN_KEEP_A", "AUGC" * 30),
        ("TRAIN_KEEP_B", "GCAU" * 30),
    ])
    holdout_csv = tmp_path / "holdout.csv"
    holdout_csv.write_text(
        "target_id,sequence\nHOLDOUT_A," + shared + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "holdout_manifest.json"

    with (
        patch(
            "rna_scaffold_3d.splitting._candidate_pairs",
            side_effect=AssertionError("external holdout must be exhaustive"),
        ),
        patch(
            "rna_scaffold_3d.splitting._anchor_candidate_pairs",
            side_effect=AssertionError("external holdout must be exhaustive"),
        ),
    ):
        filtered, audit = exclude_external_holdout(
            dataset,
            holdout_csv,
            jaccard_threshold=0.8,
            manifest_path=manifest,
        )
    remaining = {record.target_id for record in filtered.records}
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert remaining == {"TRAIN_KEEP_A", "TRAIN_KEEP_B"}
    assert audit.excluded_records == 2
    assert audit.exact_sequence_exclusions == 1
    assert audit.near_duplicate_exclusions == 1
    assert audit.cross_pairs_checked == 4
    assert audit.cross_pair_audit_exhaustive is True
    assert payload["format_version"] == 2
    assert (
        payload["parameters"]["candidate_strategy"]
        == "exhaustive_cross_product"
    )
    assert len(payload["exclusions"]) == 2
