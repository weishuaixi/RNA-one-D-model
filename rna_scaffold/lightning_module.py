from __future__ import annotations

import torch

try:
    import lightning.pytorch as L
except ImportError:  # pragma: no cover
    import pytorch_lightning as L

from rna_scaffold.model import MotifDenoisingTransformer, ScaffoldModelOutput, compute_denoising_losses


class RnaScaffoldLitModule(L.LightningModule):
    """Lightning training wrapper for motif-protected scaffold denoising."""

    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_length: int = 512,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        length_loss_weight: float = 0.25,
        position_loss_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.length_loss_weight = length_loss_weight
        self.position_loss_weight = position_loss_weight
        self.model = MotifDenoisingTransformer(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_length=max_length,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> ScaffoldModelOutput:
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> dict[str, torch.Tensor]:
        output = self(batch["input_ids"], batch["attention_mask"])
        losses = compute_denoising_losses(
            output=output,
            target_base_ids=batch["target_base_ids"],
            fixed_mask=batch["fixed_mask"],
            attention_mask=batch["attention_mask"],
            target_length=batch["target_length"],
            motif_start=batch["motif_start"],
            length_loss_weight=self.length_loss_weight,
            position_loss_weight=self.position_loss_weight,
        )
        scaffold_mask = batch["attention_mask"].bool() & ~batch["fixed_mask"].bool()
        token_accuracy = (
            output.token_logits.argmax(dim=-1)[scaffold_mask]
            == batch["target_base_ids"][scaffold_mask]
        ).float().mean()
        values = {
            "loss": losses.total_loss,
            "base_loss": losses.base_loss,
            "length_loss": losses.length_loss,
            "position_loss": losses.position_loss,
            "token_accuracy": token_accuracy,
        }
        for name, value in values.items():
            self.log(f"{stage}/{name}", value, prog_bar=name in {"loss", "token_accuracy"}, sync_dist=True)
        return values

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        return self._step(batch, "val")

    def on_train_epoch_start(self) -> None:
        datamodule = getattr(self.trainer, "datamodule", None)
        dataset = getattr(datamodule, "train_dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
