from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from rna_scaffold_3d.data import StanfordRna3DDataset, collate_3d_batch
from rna_scaffold_3d.losses import masked_coordinate_mse, masked_pairwise_distance_mse
from rna_scaffold_3d.model import Rna3DCoordinatePredictor


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small RNA sequence-to-3D coordinate predictor.")
    parser.add_argument("--config", default="configs/train_3d_a800_card1.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg.get("seed", 42))
    dataset = StanfordRna3DDataset.from_csv(**cfg["data"])
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
    model = Rna3DCoordinatePredictor(**cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"].get("weight_decay", 0.0))
    pairwise_weight = float(cfg["optimizer"].get("pairwise_weight", 0.1))

    checkpoint_dir = Path(cfg["trainer"].get("checkpoint_dir", "checkpoints_3d"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    show_progress = progress_enabled(cfg["trainer"])
    for epoch in range(int(cfg["trainer"]["max_epochs"])):
        train_loss = _run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            pairwise_weight,
            epoch=epoch + 1,
            phase="train",
            show_progress=show_progress,
        )
        val_loss = _run_epoch(
            model,
            val_loader,
            device,
            None,
            pairwise_weight,
            epoch=epoch + 1,
            phase="val",
            show_progress=show_progress,
        )
        print(f"epoch={epoch + 1} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"model_state_dict": model.state_dict(), "config": cfg, "val_loss": val_loss},
                checkpoint_dir / "rna_3d_best.pt",
            )


def _run_epoch(
    model: Rna3DCoordinatePredictor,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    pairwise_weight: float,
    epoch: int,
    phase: str,
    show_progress: bool,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_batches = 0
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
        with torch.set_grad_enabled(training):
            pred = model(input_ids=input_ids, padding_mask=padding_mask)
            loss = masked_coordinate_mse(pred, coords, coord_mask)
            loss = loss + pairwise_weight * masked_pairwise_distance_mse(pred, coords, coord_mask)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_batches += 1
        if show_progress:
            iterator.set_postfix(
                loss=f"{float(loss.detach().cpu()):.4f}",
                avg=f"{total_loss / total_batches:.4f}",
            )
    return total_loss / max(1, total_batches)


def select_training_device(trainer_cfg: dict, cuda_available: bool | None = None) -> torch.device:
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()
    if trainer_cfg.get("accelerator") == "gpu" and cuda_available:
        cuda_device = int(trainer_cfg.get("cuda_device", 0))
        return torch.device(f"cuda:{cuda_device}")
    return torch.device("cpu")


def progress_enabled(trainer_cfg: dict) -> bool:
    return bool(trainer_cfg.get("show_progress", True))


if __name__ == "__main__":
    main()
