from __future__ import annotations

import math

import torch
from torch import nn

try:
    import lightning.pytorch as L
except ImportError:  # pragma: no cover - compatibility for older installations
    import pytorch_lightning as L


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 4096) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class RnaScaffoldLitModule(L.LightningModule):
    """Encoder-decoder Transformer for motif-conditioned variable-length scaffold targets."""

    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.pad_token_id = pad_token_id
        self.lr = lr
        self.weight_decay = weight_decay

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.output = nn.Linear(d_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    def forward(self, input_ids: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        src = input_ids.transpose(0, 1)
        tgt = decoder_input_ids.transpose(0, 1)
        src_key_padding_mask = input_ids.eq(self.pad_token_id)
        tgt_key_padding_mask = decoder_input_ids.eq(self.pad_token_id)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0), device=tgt.device)

        src_emb = self.position(self.embedding(src))
        tgt_emb = self.position(self.embedding(tgt))
        hidden = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.output(hidden).transpose(0, 1)

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> dict[str, torch.Tensor]:
        labels = batch["labels"]
        decoder_input_ids = labels[:, :-1]
        target_ids = labels[:, 1:]
        logits = self(batch["input_ids"], decoder_input_ids)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        return self._step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
