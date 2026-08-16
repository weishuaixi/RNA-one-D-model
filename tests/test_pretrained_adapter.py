from __future__ import annotations

import torch
from torch import nn

import rna_scaffold.pretrained as pretrained
from rna_scaffold.model import MotifDenoisingTransformer
from rna_scaffold.pretrained import PretrainedEncoderConfig, build_pretrained_encoder


class TinyEncoder(nn.Module):
    output_dim = 6

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(12, self.output_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids) * attention_mask.unsqueeze(-1)


def test_none_pretrained_encoder_has_no_optional_dependency():
    assert build_pretrained_encoder(PretrainedEncoderConfig(kind="none")) is None


def test_frozen_pretrained_encoder_has_no_trainable_parameters(monkeypatch):
    monkeypatch.setattr(pretrained, "load_rna_fm", lambda _: TinyEncoder())
    encoder = build_pretrained_encoder(
        PretrainedEncoderConfig(kind="rna_fm", checkpoint="fake.pt", freeze=True)
    )

    assert encoder is not None
    assert not any(parameter.requires_grad for parameter in encoder.parameters())


def test_pretrained_features_are_projected_into_generator_width():
    encoder = TinyEncoder()
    model = MotifDenoisingTransformer(
        vocab_size=12,
        pad_token_id=0,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        max_length=16,
        pretrained_encoder=encoder,
    )
    output = model(
        input_ids=torch.tensor([[3, 8, 11, 3]]),
        attention_mask=torch.ones(1, 4, dtype=torch.bool),
    )

    assert model.pretrained_projection.in_features == encoder.output_dim
    assert model.pretrained_projection.out_features == 16
    assert output.token_logits.shape == (1, 4, 4)


def test_frozen_pretrained_encoder_stays_in_eval_mode(monkeypatch):
    monkeypatch.setattr(pretrained, "load_rna_fm", lambda _: TinyEncoder())
    encoder = build_pretrained_encoder(PretrainedEncoderConfig(kind="rna_fm", freeze=True))
    model = MotifDenoisingTransformer(
        vocab_size=12,
        pad_token_id=0,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        max_length=16,
        pretrained_encoder=encoder,
    )

    model.train()

    assert encoder.training is False


def test_missing_rna_fm_explains_optional_install(monkeypatch):
    monkeypatch.setattr(pretrained, "_import_fm", lambda: (_ for _ in ()).throw(ImportError("missing")))

    try:
        pretrained.load_rna_fm(None)
    except RuntimeError as error:
        assert "pip install rna-fm" in str(error)
    else:
        raise AssertionError("missing RNA-FM must raise a useful error")
