from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from rna_scaffold_3d.data import StanfordRna3DDataset, StanfordRnaAllAtomDataset, collate_3d_batch
from rna_scaffold_3d.losses import (
    base_orientation_coordinate_loss,
    base_planarity_loss,
    bond_angle_loss,
    bond_length_loss,
    frame_aligned_point_error,
    inter_residue_geometry_loss,
    kabsch_aligned_coordinate_loss,
    local_distance_difference_loss,
    soft_lddt_loss,
    masked_coordinate_mse,
    masked_pairwise_distance_mse,
    pair_distance_cross_entropy,
    pair_orientation_cross_entropy,
    plddt_confidence_loss,
    sugar_ring_closure_loss,
    sugar_pucker_coordinate_loss,
    steric_clash_loss,
    torsion_parameter_loss,
)
from rna_scaffold_3d.geometry import apply_random_rigid_augmentation
from rna_scaffold_3d.metrics import batch_structure_metrics
from rna_scaffold_3d.rhofold import (
    RHO_FOLD_ARCHITECTURE_VERSION,
    RhoFoldConfig,
    RhoFoldModel,
)
from rna_scaffold_3d.rna_atoms import chemical_atom_mask
from rna_scaffold_3d.splitting import (
    exclude_external_holdout,
    leakage_safe_train_val_split,
)

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dataset(data_cfg: dict):
    source = data_cfg.get("source", "csv_single_atom")
    training_only = {"crop_length", "external_holdout"}
    options = {
        key: value for key, value in data_cfg.items()
        if key != "source" and key not in training_only
    }
    if source == "csv_single_atom":
        return StanfordRna3DDataset.from_csv(**options)
    if source == "cif_all_atom":
        return StanfordRnaAllAtomDataset.from_csv_and_cif(**options)
    raise ValueError("data.source must be 'csv_single_atom' or 'cif_all_atom'.")


class CroppedRnaDataset(Dataset):
    """Crop long RNAs after the leakage-safe split to bound cubic triangle cost."""

    def __init__(
        self,
        dataset,
        crop_length: int,
        random_crop: bool,
        deterministic_crops: int = 1,
    ) -> None:
        self.dataset = dataset
        self.crop_length = max(1, int(crop_length))
        self.random_crop = random_crop
        self.deterministic_crops = max(1, int(deterministic_crops))
        self.windows: list[tuple[int, int, float]] | None = None
        if not random_crop:
            self.windows = []
            for index in range(len(dataset)):
                length = int(dataset[index]["input_ids"].size(0))
                if length <= self.crop_length:
                    starts = [0]
                else:
                    max_start = length - self.crop_length
                    crop_count = max(
                        self.deterministic_crops,
                        math.ceil(length / self.crop_length),
                    )
                    starts = (
                        [max_start // 2]
                        if crop_count == 1
                        else sorted({
                            round(
                                slot * max_start
                                / (crop_count - 1)
                            )
                            for slot in range(crop_count)
                        })
                    )
                weight = 1.0 / len(starts)
                self.windows.extend(
                    (index, start, weight) for start in starts
                )

    def __len__(self) -> int:
        return (
            len(self.dataset)
            if self.windows is None
            else len(self.windows)
        )

    def __getitem__(self, index: int):
        if self.windows is None:
            item_index = index
            fixed_start = None
            example_weight = 1.0
        else:
            item_index, fixed_start, example_weight = self.windows[index]
        item = self.dataset[item_index]
        length = int(item["input_ids"].size(0))
        if length <= self.crop_length:
            uncropped = dict(item)
            uncropped["example_weight"] = example_weight
            return uncropped
        if fixed_start is None:
            start = int(torch.randint(0, length - self.crop_length + 1, ()).item())
        else:
            start = fixed_start
        stop = start + self.crop_length
        cropped = dict(item)
        cropped["sequence"] = str(item["sequence"])[start:stop]
        for key in ("input_ids", "coords", "coord_mask"):
            cropped[key] = item[key][start:stop]
        cropped["target_id"] = f"{item['target_id']}:{start}-{stop}"
        cropped["example_weight"] = example_weight
        return cropped


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - NumPy is a declared dependency
        pass
    else:
        np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def numpy_rng_state() -> dict[str, object] | None:
    """Serialize NumPy's global RNG without embedding a pickled ndarray."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - NumPy is a declared dependency
        return None
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "algorithm": algorithm,
        "keys": keys.tolist(),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def restore_numpy_rng_state(state: dict[str, object] | None) -> None:
    if state is None:
        return
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - NumPy is a declared dependency
        return
    np.random.set_state(
        (
            str(state["algorithm"]),
            np.asarray(state["keys"], dtype=np.uint32),
            int(state["position"]),
            int(state["has_gauss"]),
            float(state["cached_gaussian"]),
        )
    )


def seed_data_worker(worker_id: int) -> None:
    """Seed non-Torch RNGs from the DataLoader-assigned worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - NumPy is a declared dependency
        return
    np.random.seed(worker_seed)


def validate_config(cfg: dict) -> None:
    for section in ("data", "model", "optimizer", "trainer"):
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ValueError(f"Config section '{section}' is required.")
    model = cfg["model"]
    d_model = int(model["d_model"])
    nhead = int(model["nhead"])
    pair_dim = int(model["pair_dim"])
    pair_heads = int(model.get("pair_heads") or model.get("pair_attention_heads", 4))
    if d_model % nhead:
        raise ValueError("model.d_model must be divisible by model.nhead.")
    if pair_dim % pair_heads:
        raise ValueError("model.pair_dim must be divisible by model.pair_heads.")
    for name in ("sequence_mask_probability", "scaffold_mask_probability", "val_fraction"):
        value = float(cfg["trainer"].get(name, 0.0))
        if not 0.0 <= value < 1.0:
            raise ValueError(f"trainer.{name} must be in [0, 1).")
    if int(cfg["trainer"].get("batch_size", 0)) <= 0:
        raise ValueError("trainer.batch_size must be positive.")
    accelerator = str(cfg["trainer"].get("accelerator", "cpu")).lower()
    if accelerator not in {"cpu", "gpu"}:
        raise ValueError("trainer.accelerator must be 'cpu' or 'gpu'.")
    if int(cfg["trainer"].get("cuda_device", 0)) < 0:
        raise ValueError("trainer.cuda_device must be non-negative.")
    if int(cfg["trainer"].get("num_workers", 0)) < 0:
        raise ValueError("trainer.num_workers must be non-negative.")
    if int(cfg["trainer"].get("max_epochs", 0)) <= 0:
        raise ValueError("trainer.max_epochs must be positive.")
    if int(cfg["trainer"].get("accumulate_grad_batches", 1)) <= 0:
        raise ValueError(
            "trainer.accumulate_grad_batches must be positive."
        )
    if int(cfg["trainer"].get("validation_crops", 3)) <= 0:
        raise ValueError("trainer.validation_crops must be positive.")
    optimizer = cfg["optimizer"]
    learning_rate = float(optimizer.get("lr", 2e-4))
    minimum_learning_rate = float(optimizer.get("min_lr", 1e-6))
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("optimizer.lr must be positive.")
    if (
        not math.isfinite(minimum_learning_rate)
        or not 0.0 <= minimum_learning_rate <= learning_rate
    ):
        raise ValueError("optimizer.min_lr must be in [0, optimizer.lr].")
    weight_decay = float(optimizer.get("weight_decay", 0.0))
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("optimizer.weight_decay must be non-negative.")
    if int(optimizer.get("warmup_steps", 0)) < 0:
        raise ValueError("optimizer.warmup_steps must be non-negative.")
    loss_weight_defaults = {
        "coord_mse_weight": 0.1,
        "raw_coord_weight": 0.0,
        "pairwise_weight": 0.1,
        "local_distance_weight": 0.5,
        "soft_lddt_weight": 0.5,
        "pair_ce_weight": 0.1,
        "orientation_weight": 0.1,
        "fape_weight": 0.05,
        "clash_weight": 0.02,
        "bond_weight": 0.05,
        "angle_weight": 0.2,
        "torsion_weight": 0.02,
        "confidence_weight": 0.01,
        "inter_residue_weight": 0.2,
        "planarity_weight": 0.05,
        "sugar_weight": 0.1,
        "pucker_weight": 0.05,
        "base_orientation_weight": 0.1,
    }
    for name, default in loss_weight_defaults.items():
        value = float(optimizer.get(name, default))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"optimizer.{name} must be non-negative.")
    fape_weight = float(optimizer.get("fape_weight", 0.05))
    if not math.isfinite(fape_weight) or fape_weight <= 0.0:
        raise ValueError(
            "optimizer.fape_weight must be positive because FAPE is the "
            "primary structure loss."
        )
    gradient_clip_norm = float(
        cfg["trainer"].get("gradient_clip_norm", 1.0)
    )
    if not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0.0:
        raise ValueError("trainer.gradient_clip_norm must be positive.")
    translation_scale = float(
        cfg["trainer"].get("random_translation_scale", 5.0)
    )
    if not math.isfinite(translation_scale) or translation_scale < 0.0:
        raise ValueError(
            "trainer.random_translation_scale must be non-negative."
        )
    split_cfg = cfg["trainer"].get("sequence_split", {})
    if not isinstance(split_cfg, dict):
        raise ValueError("trainer.sequence_split must be a mapping.")
    if not 0.0 < float(split_cfg.get("jaccard_threshold", 0.8)) <= 1.0:
        raise ValueError("trainer.sequence_split.jaccard_threshold must be in (0, 1].")
    if int(split_cfg.get("kmer_size", 8)) <= 0:
        raise ValueError("trainer.sequence_split.kmer_size must be positive.")
    holdout_cfg = cfg["data"].get("external_holdout")
    if holdout_cfg is not None:
        if not isinstance(holdout_cfg, dict) or not holdout_cfg.get("sequences_csv"):
            raise ValueError(
                "data.external_holdout must define sequences_csv."
            )
        threshold = float(holdout_cfg.get("jaccard_threshold", 0.8))
        if not 0.0 < threshold <= 1.0:
            raise ValueError(
                "data.external_holdout.jaccard_threshold must be in (0, 1]."
            )
        if int(holdout_cfg.get("kmer_size", 8)) <= 0:
            raise ValueError(
                "data.external_holdout.kmer_size must be positive."
            )


def save_training_checkpoint(
    path: str | Path,
    model: RhoFoldModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg: dict,
    *,
    epoch: int,
    val_loss: float,
    best_val: float,
    data_generators: dict[str, torch.Generator] | None = None,
    training_semantics: str | None = None,
    training_provenance: dict[str, str | None] | None = None,
) -> None:
    model_uses_cuda = any(parameter.is_cuda for parameter in model.parameters())
    torch.save(
        {
            "format_version": 5,
            "architecture_version": RHO_FOLD_ARCHITECTURE_VERSION,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": cfg,
            "val_loss": val_loss,
            "best_val": best_val,
            "checkpoint_metric": str(
                cfg.get("trainer", {}).get(
                    "checkpoint_metric", "c1_lddt"
                )
            ),
            "checkpoint_mode": str(
                cfg.get("trainer", {}).get("checkpoint_mode", "max")
            ),
            "training_semantics": training_semantics,
            "training_provenance": training_provenance,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": numpy_rng_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if model_uses_cuda else None,
            "data_generator_states": (
                {
                    name: generator.get_state()
                    for name, generator in data_generators.items()
                }
                if data_generators is not None
                else None
            ),
        },
        Path(path),
    )


def restore_training_checkpoint(
    path: str | Path,
    model: RhoFoldModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    data_generators: dict[str, torch.Generator] | None = None,
    expected_checkpoint_metric: str | None = None,
    expected_checkpoint_mode: str | None = None,
    expected_training_semantics: str | None = None,
    expected_training_provenance: dict[str, str | None] | None = None,
) -> tuple[int, float]:
    checkpoint = torch.load(str(path), map_location=device)
    if checkpoint.get("architecture_version") != RHO_FOLD_ARCHITECTURE_VERSION:
        raise ValueError(
            f"Unsupported checkpoint architecture: {checkpoint.get('architecture_version')!r}. "
            "The current model requires CCD geometry and glycosidic-axis chi; retrain it."
        )
    if "structure_module.atom_templates" not in checkpoint["model_state_dict"]:
        raise ValueError(
            "Checkpoint predates the base-specific RNA internal-coordinate templates; "
            "resume requires retraining with the current architecture."
        )
    saved_trainer = checkpoint.get("config", {}).get("trainer", {})
    saved_metric = str(
        checkpoint.get(
            "checkpoint_metric",
            saved_trainer.get("checkpoint_metric", "c1_lddt"),
        )
    )
    saved_mode = str(
        checkpoint.get(
            "checkpoint_mode",
            saved_trainer.get("checkpoint_mode", "max"),
        )
    )
    if (
        expected_checkpoint_metric is not None
        and saved_metric != expected_checkpoint_metric
    ):
        raise ValueError(
            "Cannot resume with a different checkpoint metric: "
            f"saved={saved_metric!r}, current={expected_checkpoint_metric!r}."
        )
    if (
        expected_checkpoint_mode is not None
        and saved_mode != expected_checkpoint_mode
    ):
        raise ValueError(
            "Cannot resume with a different checkpoint mode: "
            f"saved={saved_mode!r}, current={expected_checkpoint_mode!r}."
        )
    if expected_training_semantics is not None:
        saved_semantics = checkpoint.get("training_semantics")
        if saved_semantics is None:
            raise ValueError(
                "Checkpoint predates reproducible training-semantics "
                "validation; restart training with the current format."
            )
        if saved_semantics != expected_training_semantics:
            raise ValueError(
                "Cannot resume because model/data/loss/schedule semantics "
                "differ from the saved training run."
            )
    if expected_training_provenance is not None:
        saved_provenance = checkpoint.get("training_provenance")
        if saved_provenance != expected_training_provenance:
            raise ValueError(
                "Cannot resume because split/holdout artifact provenance "
                "differs from the saved training run."
            )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if "optimizer_state_dict" not in checkpoint or "scheduler_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is inference-only and cannot resume optimizer state.")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    restore_numpy_rng_state(checkpoint.get("numpy_rng_state"))
    if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    if data_generators is not None:
        states = checkpoint.get("data_generator_states")
        if not isinstance(states, dict):
            raise ValueError(
                "Checkpoint predates reproducible DataLoader RNG state; "
                "restart training with the current checkpoint format."
            )
        missing_generators = sorted(set(data_generators) - set(states))
        if missing_generators:
            raise ValueError(
                "Checkpoint is missing DataLoader RNG states: "
                + ", ".join(missing_generators)
            )
        for name, generator in data_generators.items():
            generator.set_state(states[name].cpu())
    return int(checkpoint.get("epoch", 0)), float(checkpoint.get("best_val", float("inf")))


def checkpoint_selection(
    metrics: dict[str, float],
    metric_name: str,
    mode: str,
    best_value: float,
) -> tuple[float, bool, bool]:
    """Return selection value, validity, and whether it improves the best."""
    if metric_name not in metrics:
        raise ValueError(
            f"Unknown trainer.checkpoint_metric {metric_name!r}; "
            f"available metrics: {', '.join(sorted(metrics))}."
        )
    value = float(metrics[metric_name])
    count = float(metrics.get(f"{metric_name}_count", 1.0))
    valid = math.isfinite(value) and math.isfinite(count) and count > 0
    improved = valid and (
        value > best_value if mode == "max" else value < best_value
    )
    return value, valid, improved


def _semantic_config(cfg: dict) -> dict:
    """Return path/device/logging-independent configuration semantics."""
    data = dict(cfg.get("data", {}))
    for key in ("sequences_csv", "cif_dir", "cache_path"):
        data.pop(key, None)
    if isinstance(data.get("external_holdout"), dict):
        holdout = dict(data["external_holdout"])
        holdout.pop("sequences_csv", None)
        holdout.pop("manifest_path", None)
        data["external_holdout"] = holdout

    trainer = dict(cfg.get("trainer", {}))
    for key in (
        "cuda_device",
        "pin_memory",
        "checkpoint_dir",
        "show_progress",
    ):
        trainer.pop(key, None)
    if isinstance(trainer.get("sequence_split"), dict):
        split = dict(trainer["sequence_split"])
        split.pop("manifest_path", None)
        trainer["sequence_split"] = split
    return {
        "seed": int(cfg.get("seed", 42)),
        "data": data,
        "model": cfg.get("model", {}),
        "optimizer": cfg.get("optimizer", {}),
        "trainer": trainer,
    }


def _record_label_fingerprint(record) -> str:
    digest = hashlib.sha256()
    for name in ("coords", "coord_mask"):
        tensor = getattr(record, name, None)
        if not isinstance(tensor, torch.Tensor):
            digest.update(f"{name}:missing".encode("ascii"))
            continue
        contiguous = tensor.detach().cpu().contiguous().clone()
        digest.update(name.encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(bytes(contiguous.untyped_storage()))
    return digest.hexdigest()


def dataset_membership(dataset) -> list[tuple[str, str, str]]:
    """Return ordered target, sequence, and exact-label identities."""
    if isinstance(dataset, Subset):
        records = getattr(dataset.dataset, "records", None)
        if records is None:
            raise ValueError("Subset base dataset must expose records.")
        return [
            (
                str(records[index].target_id),
                str(records[index].sequence),
                _record_label_fingerprint(records[index]),
            )
            for index in dataset.indices
        ]
    records = getattr(dataset, "records", None)
    if records is None:
        raise ValueError("Training dataset must expose records.")
    return [
        (
            str(record.target_id),
            str(record.sequence),
            _record_label_fingerprint(record),
        )
        for record in records
    ]


def training_semantics_fingerprint(
    cfg: dict,
    train_members: list[tuple[str, ...]],
    val_members: list[tuple[str, ...]],
    total_steps: int,
) -> str:
    payload = {
        "format_version": 7,
        "config": _semantic_config(cfg),
        "train_members": train_members,
        "val_members": val_members,
        "total_steps": int(total_steps),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_artifact_provenance(
    cfg: dict,
) -> dict[str, str | None]:
    """Hash the exact leakage manifests consumed by a training run."""
    split = cfg.get("trainer", {}).get("sequence_split", {})
    holdout = cfg.get("data", {}).get("external_holdout", {})
    paths = {
        "split_manifest_sha256": (
            split.get("manifest_path") if isinstance(split, dict) else None
        ),
        "holdout_manifest_sha256": (
            holdout.get("manifest_path")
            if isinstance(holdout, dict)
            else None
        ),
    }
    return {
        name: _sha256_file(path)
        if path and Path(path).is_file()
        else None
        for name, path in paths.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small RNA sequence-to-3D coordinate predictor.")
    parser.add_argument("--config", default="configs/train_3d_a800_card1.yaml")
    parser.add_argument("--resume", help="Resume model, optimizer, scheduler and epoch state.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    seed_everything(int(cfg.get("seed", 42)))
    validate_config(cfg)
    cfg["model"] = normalize_rhofold_config(cfg["model"])
    device = select_training_device(
        cfg["trainer"],
        cuda_available=torch.cuda.is_available(),
        cuda_device_count=torch.cuda.device_count(),
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
        print(
            f"Using CUDA device {device.index}: "
            f"{torch.cuda.get_device_name(device)}"
        )
    else:
        print("Using CPU")
    dataset = build_dataset(cfg["data"])
    if not dataset:
        raise ValueError("No RNA 3D training records were loaded.")
    holdout_cfg = cfg["data"].get("external_holdout")
    if holdout_cfg:
        dataset, holdout_audit = exclude_external_holdout(
            dataset,
            holdout_sequences_csv=holdout_cfg["sequences_csv"],
            jaccard_threshold=float(
                holdout_cfg.get("jaccard_threshold", 0.8)
            ),
            kmer_size=int(holdout_cfg.get("kmer_size", 8)),
            manifest_path=holdout_cfg.get("manifest_path"),
        )
        print("holdout_audit=" + ", ".join(
            f"{name}={value}"
            for name, value in holdout_audit.as_dict().items()
        ))
    if hasattr(dataset, "stats"):
        print("data_stats=" + ", ".join(
            f"{name}={value}" for name, value in sorted(dataset.stats.items())
        ))

    split_cfg = cfg["trainer"].get("sequence_split", {})
    train_dataset, val_dataset, split_audit = leakage_safe_train_val_split(
        dataset,
        val_fraction=float(cfg["trainer"].get("val_fraction", 0.05)),
        seed=int(cfg.get("seed", 42)),
        manifest_path=split_cfg.get("manifest_path"),
        kmer_size=int(split_cfg.get("kmer_size", 8)),
        jaccard_threshold=float(split_cfg.get("jaccard_threshold", 0.8)),
    )
    print("split_audit=" + ", ".join(
        f"{name}={value}" for name, value in split_audit.as_dict().items()
    ))
    train_members = dataset_membership(train_dataset)
    val_members = dataset_membership(val_dataset)
    crop_length = cfg["data"].get("crop_length")
    validation_crops = int(cfg["trainer"].get("validation_crops", 3))
    if crop_length:
        train_dataset = CroppedRnaDataset(train_dataset, crop_length, random_crop=True)
        val_dataset = CroppedRnaDataset(
            val_dataset,
            crop_length,
            random_crop=False,
            deterministic_crops=validation_crops,
        )

    seed = int(cfg.get("seed", 42))
    data_generators = {
        "train": torch.Generator().manual_seed(seed + 10_001),
        "val": torch.Generator().manual_seed(seed + 20_003),
    }
    num_workers = int(cfg["trainer"].get("num_workers", 0))
    persistent_workers = bool(
        cfg["trainer"].get("persistent_workers", False)
    ) and num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["trainer"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_3d_batch,
        pin_memory=bool(cfg["trainer"].get("pin_memory", True)),
        persistent_workers=persistent_workers,
        worker_init_fn=seed_data_worker,
        generator=data_generators["train"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=(
            1
            if crop_length and validation_crops > 1
            else cfg["trainer"]["batch_size"]
        ),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_3d_batch,
        pin_memory=bool(cfg["trainer"].get("pin_memory", True)),
        persistent_workers=persistent_workers,
        worker_init_fn=seed_data_worker,
        generator=data_generators["val"],
    )

    model = RhoFoldModel(RhoFoldConfig(**cfg["model"])).to(device)
    task_weight_parameters = [model.task_log_variances]
    network_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name != "task_log_variances"
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": network_parameters,
                "weight_decay": cfg["optimizer"].get("weight_decay", 0.0),
            },
            {"params": task_weight_parameters, "weight_decay": 0.0},
        ],
        lr=cfg["optimizer"]["lr"],
    )
    accumulate_grad_batches = int(cfg["trainer"].get("accumulate_grad_batches", 1))
    total_steps = max(1, (len(train_loader) + accumulate_grad_batches - 1) // accumulate_grad_batches) * int(cfg["trainer"]["max_epochs"])
    training_semantics = training_semantics_fingerprint(
        cfg, train_members, val_members, total_steps
    )
    training_provenance = training_artifact_provenance(cfg)
    scheduler = build_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=int(cfg["optimizer"].get("warmup_steps", max(1, total_steps // 20))),
        min_lr_ratio=float(cfg["optimizer"].get("min_lr", 1e-6)) / float(cfg["optimizer"]["lr"]),
    )
    pairwise_weight = float(cfg["optimizer"].get("pairwise_weight", 0.1))
    local_distance_weight = float(cfg["optimizer"].get("local_distance_weight", 0.5))
    soft_lddt_weight = float(cfg["optimizer"].get("soft_lddt_weight", 0.5))
    pair_ce_weight = float(cfg["optimizer"].get("pair_ce_weight", 0.1))
    orientation_weight = float(cfg["optimizer"].get("orientation_weight", 0.1))
    coord_mse_weight = float(cfg["optimizer"].get("coord_mse_weight", 0.1))
    raw_coord_weight = float(cfg["optimizer"].get("raw_coord_weight", 0.0))
    fape_weight = float(cfg["optimizer"].get("fape_weight", 0.05))
    clash_weight = float(cfg["optimizer"].get("clash_weight", 0.02))
    bond_weight = float(cfg["optimizer"].get("bond_weight", 0.05))
    angle_weight = float(cfg["optimizer"].get("angle_weight", 0.2))
    torsion_weight = float(cfg["optimizer"].get("torsion_weight", 0.02))
    confidence_weight = float(cfg["optimizer"].get("confidence_weight", 0.01))
    inter_residue_weight = float(cfg["optimizer"].get("inter_residue_weight", 0.2))
    planarity_weight = float(cfg["optimizer"].get("planarity_weight", 0.05))
    sugar_weight = float(cfg["optimizer"].get("sugar_weight", 0.1))
    pucker_weight = float(cfg["optimizer"].get("pucker_weight", 0.05))
    base_orientation_weight = float(
        cfg["optimizer"].get("base_orientation_weight", 0.1)
    )

    checkpoint_dir = Path(cfg["trainer"].get("checkpoint_dir", "checkpoints_3d"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_metric = str(cfg["trainer"].get("checkpoint_metric", "c1_lddt"))
    checkpoint_mode = str(cfg["trainer"].get("checkpoint_mode", "max"))
    if checkpoint_mode not in {"min", "max"}:
        raise ValueError("trainer.checkpoint_mode must be 'min' or 'max'.")
    best_val = float("-inf") if checkpoint_mode == "max" else float("inf")
    start_epoch = 0
    if args.resume:
        start_epoch, best_val = restore_training_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            device,
            data_generators=data_generators,
            expected_checkpoint_metric=checkpoint_metric,
            expected_checkpoint_mode=checkpoint_mode,
            expected_training_semantics=training_semantics,
            expected_training_provenance=training_provenance,
        )
    show_progress = progress_enabled(cfg["trainer"])
    wandb_run = init_wandb(cfg)
    for epoch in range(start_epoch, int(cfg["trainer"]["max_epochs"])):
        loss_weights = {
            "pairwise": pairwise_weight,
            "local_distance": local_distance_weight,
            "soft_lddt": soft_lddt_weight,
            "pair_ce": pair_ce_weight,
            "orientation": orientation_weight,
            "coord_mse": coord_mse_weight,
            "raw_coord": raw_coord_weight,
            "fape": fape_weight,
            "clash": clash_weight,
            "bond": bond_weight,
            "angle": angle_weight,
            "torsion": torsion_weight,
            "confidence": confidence_weight,
            "inter_residue": inter_residue_weight,
            "planarity": planarity_weight,
            "sugar": sugar_weight,
            "pucker": pucker_weight,
            "base_orientation": base_orientation_weight,
        }
        train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scheduler,
            loss_weights,
            trainer_cfg=cfg["trainer"],
            epoch=epoch + 1,
            phase="train",
            show_progress=show_progress,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            device,
            None,
            None,
            loss_weights,
            trainer_cfg=cfg["trainer"],
            epoch=epoch + 1,
            phase="val",
            show_progress=show_progress,
        )
        train_loss = train_metrics["loss"]
        val_loss = val_metrics["loss"]
        selection_value, selection_valid, improved = checkpoint_selection(
            val_metrics, checkpoint_metric, checkpoint_mode, best_val
        )
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"val_kabsch={val_metrics['kabsch_rmsd']:.3f} "
            f"val_lddt={val_metrics['c1_lddt']:.2f} "
            f"val_adj_c1={val_metrics['adjacent_c1_mean']:.3f}"
        )
        metrics = {
            "epoch": epoch + 1,
            "checkpoint/selection_value": selection_value,
            "checkpoint/selection_best": (
                (
                    max(best_val, selection_value)
                    if checkpoint_mode == "max"
                    else min(best_val, selection_value)
                )
                if selection_valid
                else best_val
            ),
            "optimizer/lr": optimizer.param_groups[0]["lr"],
        }
        metrics.update({f"train/{name}": value for name, value in train_metrics.items()})
        metrics.update({f"val/{name}": value for name, value in val_metrics.items()})
        structure_weight, sequence_weight = model.learned_task_weights()
        metrics["loss_weight/structure"] = float(structure_weight.detach().cpu())
        metrics["loss_weight/sequence"] = float(sequence_weight.detach().cpu())
        checkpoint_path: Path | None = None
        if improved:
            best_val = selection_value
            checkpoint_path = checkpoint_dir / "rna_3d_best.pt"
            metrics["checkpoint/best_path"] = str(checkpoint_path)
        elif not selection_valid:
            print(
                f"checkpoint_skip=metric {checkpoint_metric!r} has no "
                "finite, valid examples"
            )
        if wandb_run is not None:
            wandb_run.log(metrics, step=epoch + 1)
        if checkpoint_path is not None:
            save_training_checkpoint(
                checkpoint_path, model, optimizer, scheduler, cfg,
                epoch=epoch + 1, val_loss=val_loss, best_val=best_val,
                data_generators=data_generators,
                training_semantics=training_semantics,
                training_provenance=training_provenance,
            )
        save_training_checkpoint(
            checkpoint_dir / "rna_3d_last.pt", model, optimizer, scheduler, cfg,
            epoch=epoch + 1, val_loss=val_loss, best_val=best_val,
            data_generators=data_generators,
            training_semantics=training_semantics,
            training_provenance=training_provenance,
        )
    if wandb_run is not None:
        wandb_run.finish()


def _run_epoch(
    model: RhoFoldModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    loss_weights: dict[str, float],
    trainer_cfg: dict,
    epoch: int,
    phase: str,
    show_progress: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    denominators: dict[str, float] = {}
    metric_counts: dict[str, float] = {}
    total_example_weight = 0.0
    total_batches = 0
    accumulated_examples = 0
    accumulation = max(1, int(trainer_cfg.get("accumulate_grad_batches", 1)))
    use_amp = mixed_precision_enabled(trainer_cfg, device)
    # The configured autocast dtype is bfloat16, whose exponent range does not
    # need FP16-style gradient scaling. Keeping a disabled scaler preserves one
    # optimizer path for CPU/FP32 and CUDA/BF16.
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    if training:
        optimizer.zero_grad(set_to_none=True)
    iterator = tqdm(
        loader,
        desc=f"epoch {epoch} {phase}",
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for batch in iterator:
        target_input_ids = batch["input_ids"].to(device)
        batch_examples = int(target_input_ids.size(0))
        batch_example_weight = float(
            batch.get(
                "example_weights",
                torch.ones(batch_examples),
            ).sum().item()
        )
        coords = batch["coords"].to(device)
        coord_mask = batch["coord_mask"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        physical_atom_mask = chemical_atom_mask(target_input_ids)
        input_ids, sequence_mask = mask_sequence_inputs(
            target_input_ids,
            padding_mask,
            mask_token_id=model.config.vocab_size - 1,
            mask_probability=float(trainer_cfg.get("sequence_mask_probability", 0.15)),
            scaffold_mask_probability=float(trainer_cfg.get("scaffold_mask_probability", 0.25)),
            motif_length=int(trainer_cfg.get("motif_length", 16)),
            training=True,
            deterministic=not training,
        )
        if training and bool(trainer_cfg.get("random_rotation_augmentation", True)):
            coords = apply_random_rigid_augmentation(
                coords,
                coord_mask,
                translation_scale=float(trainer_cfg.get("random_translation_scale", 5.0)),
            )
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            output = model(input_ids=input_ids, padding_mask=padding_mask, return_aux=True)
            pred = output["coords"]
            frame_loss = frame_aligned_point_error(
                pred, coords, coord_mask, input_ids=target_input_ids
            )
            aligned_coord_loss = kabsch_aligned_coordinate_loss(pred, coords, coord_mask)
            coord_mse_loss = masked_coordinate_mse(pred, coords, coord_mask)
            pairwise_loss = masked_pairwise_distance_mse(pred, coords, coord_mask)
            local_distance_loss = local_distance_difference_loss(pred, coords, coord_mask)
            differentiable_lddt_loss = soft_lddt_loss(
                pred, coords, coord_mask
            )
            pair_ce_loss = pair_distance_cross_entropy(output["pair_distance_logits"], coords, coord_mask)
            orientation_loss = pair_orientation_cross_entropy(
                output["orientation_logits"],
                coords,
                coord_mask,
                input_ids=target_input_ids,
            )
            clash_loss = steric_clash_loss(
                pred, physical_atom_mask, target_input_ids
            )
            bond_loss = bond_length_loss(
                pred, physical_atom_mask, target_input_ids
            )
            angle_loss = bond_angle_loss(
                pred, physical_atom_mask, target_input_ids
            )
            torsion_loss = torsion_parameter_loss(
                output["torsions"], coords, coord_mask, target_input_ids
            )
            inter_residue_loss = inter_residue_geometry_loss(
                pred, physical_atom_mask
            )
            planarity_loss = base_planarity_loss(pred, physical_atom_mask)
            sugar_loss = sugar_ring_closure_loss(pred, physical_atom_mask)
            pucker_loss = sugar_pucker_coordinate_loss(
                pred, coords, coord_mask
            )
            base_orientation_loss = base_orientation_coordinate_loss(
                pred, coords, coord_mask, target_input_ids
            )
            confidence_loss = plddt_confidence_loss(output["plddt"], pred, coords, coord_mask)
            sequence_loss = sequence_reconstruction_loss(
                output["sequence_logits"],
                target_input_ids,
                sequence_mask,
            )
            structure_loss = loss_weights["fape"] * frame_loss
            structure_loss = structure_loss + loss_weights["coord_mse"] * aligned_coord_loss
            # Raw coordinate MSE is optional and disabled by default because it is
            # not invariant to the arbitrary global target frame.
            structure_loss = structure_loss + loss_weights["raw_coord"] * coord_mse_loss
            structure_loss = structure_loss + loss_weights["pairwise"] * pairwise_loss
            structure_loss = structure_loss + loss_weights["local_distance"] * local_distance_loss
            structure_loss = structure_loss + loss_weights["soft_lddt"] * differentiable_lddt_loss
            structure_loss = structure_loss + loss_weights["pair_ce"] * pair_ce_loss
            structure_loss = structure_loss + loss_weights["orientation"] * orientation_loss
            structure_loss = structure_loss + loss_weights["clash"] * clash_loss
            structure_loss = structure_loss + loss_weights["bond"] * bond_loss
            structure_loss = structure_loss + loss_weights["angle"] * angle_loss
            structure_loss = structure_loss + loss_weights["torsion"] * torsion_loss
            structure_loss = structure_loss + loss_weights["confidence"] * confidence_loss
            structure_loss = structure_loss + loss_weights["inter_residue"] * inter_residue_loss
            structure_loss = structure_loss + loss_weights["planarity"] * planarity_loss
            structure_loss = structure_loss + loss_weights["sugar"] * sugar_loss
            structure_loss = structure_loss + loss_weights["pucker"] * pucker_loss
            structure_loss = (
                structure_loss
                + loss_weights["base_orientation"]
                * base_orientation_loss
            )
            loss = model.combine_task_losses(structure_loss, sequence_loss)
            if training:
                scaled_loss = loss * batch_examples
                scaler.scale(scaled_loss).backward()
                accumulated_examples += batch_examples
                should_step = (total_batches + 1) % accumulation == 0
                if should_step:
                    scaler.unscale_(optimizer)
                    normalize_accumulated_gradients(
                        model, accumulated_examples
                    )
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(trainer_cfg.get("gradient_clip_norm", 1.0)))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_examples = 0
                    if scheduler is not None:
                        scheduler.step()
        with torch.no_grad():
            geometry_metrics = batch_structure_metrics(pred, coords, coord_mask)
            batch_metrics = {
                "loss": loss,
                "structure_loss": structure_loss,
                "sequence_loss": sequence_loss,
                "fape": frame_loss,
                "aligned_coordinate": aligned_coord_loss,
                "raw_coordinate": coord_mse_loss,
                "pairwise_distance": pairwise_loss,
                "soft_lddt_loss": differentiable_lddt_loss,
                "distogram": pair_ce_loss,
                "orientation": orientation_loss,
                "clash": clash_loss,
                "bond": bond_loss,
                "angle": angle_loss,
                "torsion": torsion_loss,
                "confidence": confidence_loss,
                "inter_residue": inter_residue_loss,
                "planarity": planarity_loss,
                "sugar": sugar_loss,
                "pucker": pucker_loss,
                "base_orientation": base_orientation_loss,
                **geometry_metrics,
            }
        batch_valid_counts = {
            name.removesuffix("_count"): float(value.detach().cpu())
            for name, value in batch_metrics.items()
            if name.endswith("_count")
        }
        for name, value in batch_metrics.items():
            if name.endswith("_count"):
                continue
            if name in batch_valid_counts:
                weight = (
                    batch_valid_counts[name]
                    * batch_example_weight
                    / max(1, batch_examples)
                )
            else:
                weight = batch_example_weight
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * weight
            denominators[name] = denominators.get(name, 0.0) + weight
        for name, count in batch_valid_counts.items():
            metric_counts[name] = metric_counts.get(name, 0.0) + (
                count * batch_example_weight / max(1, batch_examples)
            )
        total_example_weight += batch_example_weight
        total_batches += 1
        if show_progress:
            iterator.set_postfix(
                loss=f"{float(loss.detach().cpu()):.4f}",
                avg=f"{totals['loss'] / max(1.0, total_example_weight):.4f}",
                lddt=f"{float(geometry_metrics['c1_lddt']):.1f}",
            )
    if training and total_batches % accumulation != 0:
        scaler.unscale_(optimizer)
        normalize_accumulated_gradients(model, accumulated_examples)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(trainer_cfg.get("gradient_clip_norm", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
    result = {
        name: total / max(
            1.0, denominators.get(name, total_example_weight)
        )
        for name, total in totals.items()
    }
    result.update({
        f"{name}_count": count for name, count in metric_counts.items()
    })
    return result


def normalize_accumulated_gradients(
    model: torch.nn.Module,
    example_count: int,
) -> None:
    """Convert summed per-example gradients to an exact example mean."""
    if example_count <= 0:
        raise ValueError("example_count must be positive.")
    inverse_count = 1.0 / float(example_count)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(inverse_count)


def mask_sequence_inputs(
    input_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    mask_token_id: int,
    mask_probability: float,
    scaffold_mask_probability: float,
    motif_length: int,
    training: bool,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = ~padding_mask
    if not training:
        return input_ids, torch.zeros_like(valid)

    masked = input_ids.clone()
    if deterministic:
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        row_signature = (input_ids * (positions + 11)).sum(dim=1, keepdim=True)
        scores = (positions * 37 + input_ids * 17 + row_signature) % 1009
        selected = (scores < round(mask_probability * 1009)) & valid
    else:
        selected = (torch.rand_like(input_ids, dtype=torch.float32) < mask_probability) & valid
    for row in range(input_ids.size(0)):
        valid_length = int(valid[row].sum().item())
        if valid_length <= 0:
            continue
        if valid_length == 1:
            selected[row, valid[row].nonzero(as_tuple=False)[0, 0]] = True
            continue
        scaffold_selected = (
            int((input_ids[row] * torch.arange(1, input_ids.size(1) + 1, device=input_ids.device)).sum().item()) % 1000
            < round(scaffold_mask_probability * 1000)
            if deterministic
            else torch.rand((), device=input_ids.device).item() < scaffold_mask_probability
        )
        if scaffold_selected:
            keep_length = min(max(1, motif_length), valid_length)
            if deterministic:
                span = valid_length - keep_length + 1
                start = int(input_ids[row, :valid_length].sum().item()) % span
            else:
                start = int(torch.randint(0, valid_length - keep_length + 1, (), device=input_ids.device).item())
            selected[row, :valid_length] = True
            selected[row, start : start + keep_length] = False
        elif not selected[row, :valid_length].any():
            position = (
                int(input_ids[row, :valid_length].sum().item()) % valid_length
                if deterministic
                else int(torch.randint(0, valid_length, (), device=input_ids.device).item())
            )
            selected[row, position] = True
    masked[selected] = mask_token_id
    return masked, selected


def sequence_reconstruction_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    if not selected.any():
        return logits.sum() * 0.0
    token_loss = F.cross_entropy(
        logits.transpose(1, 2),
        targets,
        reduction="none",
    )
    selected_count = selected.sum(dim=1)
    per_example = (
        (token_loss * selected).sum(dim=1)
        / selected_count.clamp(min=1).to(token_loss.dtype)
    )
    valid_examples = selected_count.gt(0)
    return per_example[valid_examples].mean()


def select_training_device(
    trainer_cfg: dict,
    cuda_available: bool | None = None,
    cuda_device_count: int | None = None,
) -> torch.device:
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()
    if trainer_cfg.get("accelerator") == "gpu":
        if not cuda_available:
            raise RuntimeError(
                "trainer.accelerator='gpu' was requested, but CUDA is "
                "unavailable. Refusing to silently run the 3D model on CPU."
            )
        cuda_device = int(trainer_cfg.get("cuda_device", 0))
        if (
            cuda_device_count is not None
            and not 0 <= cuda_device < cuda_device_count
        ):
            raise RuntimeError(
                f"trainer.cuda_device={cuda_device} is unavailable; "
                f"detected {cuda_device_count} CUDA device(s)."
            )
        return torch.device(f"cuda:{cuda_device}")
    return torch.device("cpu")


def normalize_rhofold_config(model_cfg: dict) -> dict:
    cfg = dict(model_cfg)
    cfg.pop("type", None)
    return cfg


def progress_enabled(trainer_cfg: dict) -> bool:
    return bool(trainer_cfg.get("show_progress", True))


def mixed_precision_enabled(trainer_cfg: dict, device: torch.device) -> bool:
    return bool(trainer_cfg.get("mixed_precision", False)) and device.type == "cuda"


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    min_lr_ratio = max(0.0, min(1.0, float(min_lr_ratio)))

    def lr_lambda(step: int) -> float:
        current = step + 1
        if warmup_steps and current <= warmup_steps:
            return max(min_lr_ratio, current / (warmup_steps + 1))
        if warmup_steps and current == warmup_steps + 1:
            return 1.0
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (current - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def wandb_enabled(wandb_cfg: dict) -> bool:
    return bool(wandb_cfg.get("enabled", True))


def init_wandb(cfg: dict):
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_enabled(wandb_cfg):
        return None
    if wandb is None:
        raise ImportError("wandb is enabled but not installed. Run `pip install wandb`.")
    return wandb.init(
        project=wandb_cfg.get("project", "rna-one-d-3d"),
        name=wandb_cfg.get("name"),
        entity=wandb_cfg.get("entity"),
        config=cfg,
        mode=wandb_cfg.get("mode", "online"),
    )


if __name__ == "__main__":
    main()
