from __future__ import annotations

import argparse
from pathlib import Path

import yaml

try:
    import lightning.pytorch as L
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger
except ImportError:  # pragma: no cover
    import pytorch_lightning as L
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger

from rna_scaffold.datamodule import RnaScaffoldDataModule
from rna_scaffold.lightning_module import RnaScaffoldLitModule
from rna_scaffold.tokenizer import RnaTokenizer


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RNA motif-conditioned scaffold model.")
    parser.add_argument("--config", default="configs/train_scaffold_a800.yaml")
    parser.add_argument("--resume", help="Resume from an existing Lightning checkpoint.")
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
        auto_insert_metric_name=False,
    )
    early_stopping = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=int(cfg["trainer"].get("early_stopping_patience", 12)),
        check_finite=True,
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
        callbacks=[checkpoint, early_stopping, lr_monitor],
        **cfg["trainer"]["args"],
    )
    resume = args.resume or cfg["trainer"].get("resume_from_checkpoint")
    trainer.fit(model, datamodule=data, ckpt_path=resume)


if __name__ == "__main__":
    main()
