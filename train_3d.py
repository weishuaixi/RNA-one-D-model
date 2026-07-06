from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from rna_scaffold_3d.data import StanfordRna3DDataset, StanfordRnaAllAtomDataset, collate_3d_batch
from rna_scaffold_3d.losses import (
    bond_angle_loss,
    bond_length_loss,
    local_frame_mse,
    masked_coordinate_huber,
    masked_coordinate_mse,
    masked_pairwise_distance_mse,
    pair_distance_cross_entropy,
    plddt_confidence_loss,
    secondary_logits_bce_loss,
    secondary_structure_pair_loss,
    steric_clash_loss,
    torsion_angle_loss,
)
from rna_scaffold_3d.rhofold import RhoFoldConfig, RhoFoldModel

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dataset(data_cfg: dict):
    source = data_cfg.get("source", "csv_single_atom")
    options = {key: value for key, value in data_cfg.items() if key != "source"}
    if source == "csv_single_atom":
        return StanfordRna3DDataset.from_csv(**options)
    if source == "cif_all_atom":
        return StanfordRnaAllAtomDataset.from_csv_and_cif(**options)
    raise ValueError("data.source must be 'csv_single_atom' or 'cif_all_atom'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small RNA sequence-to-3D coordinate predictor.")
    parser.add_argument("--config", default="configs/train_3d_a800_card1.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg.get("seed", 42))
    dataset = build_dataset(cfg["data"])
    if not dataset:
        raise ValueError("No RNA 3D training records were loaded.")

    val_fraction = float(cfg["trainer"].get("val_fraction", 0.05))
    val_size = max(1, int(len(dataset) * val_fraction)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    if val_size:
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg.get("seed", 42)),
        )
    else:
        train_dataset = dataset
        val_dataset = dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["trainer"]["batch_size"],
        shuffle=True,
        num_workers=cfg["trainer"].get("num_workers", 0),
        collate_fn=collate_3d_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["trainer"]["batch_size"],
        shuffle=False,
        num_workers=cfg["trainer"].get("num_workers", 0),
        collate_fn=collate_3d_batch,
    )

    device = select_training_device(cfg["trainer"], cuda_available=torch.cuda.is_available())
    if device.type == "cuda":
        torch.cuda.set_device(device)
        print(f"Using CUDA device {device.index}: {torch.cuda.get_device_name(device)}")
    else:
        print("Using CPU")
    cfg["model"] = normalize_rhofold_config(cfg["model"])
    model = RhoFoldModel(RhoFoldConfig(**cfg["model"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"].get("weight_decay", 0.0))
    accumulate_grad_batches = int(cfg["trainer"].get("accumulate_grad_batches", 1))
    total_steps = max(1, (len(train_loader) + accumulate_grad_batches - 1) // accumulate_grad_batches) * int(cfg["trainer"]["max_epochs"])
    scheduler = build_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=int(cfg["optimizer"].get("warmup_steps", max(1, total_steps // 20))),
        min_lr_ratio=float(cfg["optimizer"].get("min_lr", 1e-6)) / float(cfg["optimizer"]["lr"]),
    )
    pairwise_weight = float(cfg["optimizer"].get("pairwise_weight", 0.1))
    pair_ce_weight = float(cfg["optimizer"].get("pair_ce_weight", 0.1))
    coord_mse_weight = float(cfg["optimizer"].get("coord_mse_weight", 0.1))
    fape_weight = float(cfg["optimizer"].get("fape_weight", 0.05))
    clash_weight = float(cfg["optimizer"].get("clash_weight", 0.02))
    bond_weight = float(cfg["optimizer"].get("bond_weight", 0.05))
    angle_weight = float(cfg["optimizer"].get("angle_weight", 0.02))
    torsion_weight = float(cfg["optimizer"].get("torsion_weight", 0.02))
    secondary_weight = float(cfg["optimizer"].get("secondary_weight", 0.02))
    confidence_weight = float(cfg["optimizer"].get("confidence_weight", 0.01))

    checkpoint_dir = Path(cfg["trainer"].get("checkpoint_dir", "checkpoints_3d"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    show_progress = progress_enabled(cfg["trainer"])
    wandb_run = init_wandb(cfg)
    for epoch in range(int(cfg["trainer"]["max_epochs"])):
        loss_weights = {
            "pairwise": pairwise_weight,
            "pair_ce": pair_ce_weight,
            "coord_mse": coord_mse_weight,
            "fape": fape_weight,
            "clash": clash_weight,
            "bond": bond_weight,
            "angle": angle_weight,
            "torsion": torsion_weight,
            "secondary": secondary_weight,
            "confidence": confidence_weight,
        }
        train_loss = _run_epoch(
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
        val_loss = _run_epoch(
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
        print(f"epoch={epoch + 1} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        metrics = {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "best/val_loss": min(best_val, val_loss),
            "optimizer/lr": optimizer.param_groups[0]["lr"],
        }
        if val_loss < best_val:
            best_val = val_loss
            checkpoint_path = checkpoint_dir / "rna_3d_best.pt"
            torch.save(
                {"model_state_dict": model.state_dict(), "config": cfg, "val_loss": val_loss},
                checkpoint_path,
            )
            metrics["checkpoint/best_path"] = str(checkpoint_path)
        if wandb_run is not None:
            wandb_run.log(metrics, step=epoch + 1)
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
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_batches = 0
    accumulation = max(1, int(trainer_cfg.get("accumulate_grad_batches", 1)))
    use_amp = mixed_precision_enabled(trainer_cfg, device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
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
        input_ids = batch["input_ids"].to(device)
        coords = batch["coords"].to(device)
        coord_mask = batch["coord_mask"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            output = model(input_ids=input_ids, padding_mask=padding_mask, return_aux=True)
            pred = output["coords"]
            coord_loss = masked_coordinate_huber(pred, coords, coord_mask)
            coord_mse_loss = masked_coordinate_mse(pred, coords, coord_mask)
            pairwise_loss = masked_pairwise_distance_mse(pred, coords, coord_mask)
            pair_ce_loss = pair_distance_cross_entropy(output["pair_distance_logits"], coords, coord_mask)
            frame_loss = local_frame_mse(pred, coords, coord_mask)
            clash_loss = steric_clash_loss(pred, coord_mask)
            bond_loss = bond_length_loss(pred, coord_mask)
            angle_loss = bond_angle_loss(pred, coord_mask)
            torsion_loss = torsion_angle_loss(pred, coord_mask)
            secondary_loss = secondary_structure_pair_loss(pred, coord_mask, input_ids)
            secondary_head_loss = secondary_logits_bce_loss(output["secondary_logits"], input_ids, padding_mask)
            confidence_loss = plddt_confidence_loss(output["plddt"], pred, coords, coord_mask)
            loss = coord_loss
            loss = loss + loss_weights["coord_mse"] * coord_mse_loss
            loss = loss + loss_weights["pairwise"] * pairwise_loss
            loss = loss + loss_weights["pair_ce"] * pair_ce_loss
            loss = loss + loss_weights["fape"] * frame_loss
            loss = loss + loss_weights["clash"] * clash_loss
            loss = loss + loss_weights["bond"] * bond_loss
            loss = loss + loss_weights["angle"] * angle_loss
            loss = loss + loss_weights["torsion"] * torsion_loss
            loss = loss + loss_weights["secondary"] * (secondary_loss + secondary_head_loss)
            loss = loss + loss_weights["confidence"] * confidence_loss
            if training:
                scaled_loss = loss / accumulation
                scaler.scale(scaled_loss).backward()
                should_step = (total_batches + 1) % accumulation == 0
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(trainer_cfg.get("gradient_clip_norm", 1.0)))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()
        total_loss += float(loss.detach().cpu())
        total_batches += 1
        if show_progress:
            iterator.set_postfix(
                loss=f"{float(loss.detach().cpu()):.4f}",
                avg=f"{total_loss / total_batches:.4f}",
            )
    if training and total_batches % accumulation != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(trainer_cfg.get("gradient_clip_norm", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
    return total_loss / max(1, total_batches)


def select_training_device(trainer_cfg: dict, cuda_available: bool | None = None) -> torch.device:
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()
    if trainer_cfg.get("accelerator") == "gpu" and cuda_available:
        cuda_device = int(trainer_cfg.get("cuda_device", 0))
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
