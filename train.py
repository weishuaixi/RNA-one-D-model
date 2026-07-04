from __future__ import annotations

import argparse
from pathlib import Path

import yaml

try:
    import lightning.pytorch as L
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger
except ImportError:  # pragma: no cover
    import pytorch_lightning as L
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger

from rna_scaffold.datamodule import RnaScaffoldDataModule
from rna_scaffold.lightning_module import RnaScaffoldLitModule
from rna_scaffold.tokenizer import RnaTokenizer


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RNA motif-conditioned scaffold model.")
    parser.add_argument("--config", default="configs/train_a800.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    L.seed_everything(cfg.get("seed", 42), workers=True)

    tokenizer = RnaTokenizer()
    data = RnaScaffoldDataModule(tokenizer=tokenizer, **cfg["data"])
    model = RnaScaffoldLitModule(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        **cfg["model"],
    )

    checkpoint = ModelCheckpoint(
        dirpath=cfg["trainer"].get("checkpoint_dir", "checkpoints"),
        filename="rna-scaffold-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = WandbLogger(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"].get("name"),
        entity=cfg["wandb"].get("entity"),
        log_model=cfg["wandb"].get("log_model", False),
        config=cfg,
    )

    trainer = L.Trainer(
        logger=logger,
        callbacks=[checkpoint, lr_monitor],
        **cfg["trainer"]["args"],
    )
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
