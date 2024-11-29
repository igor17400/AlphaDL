# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import numpy as np
import pandas as pd
import math

import torch
from torch import nn, optim
from lightning import LightningModule


# qrun examples/benchmarks/Transformer/workflow_config_transformer_Alpha360.yaml ”


class TransformerModel(LightningModule):
    def __init__(
        self,
        d_feat: int = 20,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 2,
        dropout: float = 0,
        lr=0.0001,
        loss="mse",
        optimizer="adam",
        reg=1e-3,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for easy access
        self.model = Transformer(d_feat, d_model, nhead, num_layers, dropout)
        self.loss_fn = nn.MSELoss() if loss == "mse" else None
        if self.loss_fn is None:
            raise ValueError(f"Unknown loss function: {loss}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optimizer = optim.Adam(
                self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.reg
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = optim.SGD(
                self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.reg
            )
        else:
            raise NotImplementedError(
                f"Optimizer {self.hparams.optimizer} is not supported!"
            )
        return optimizer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]


class Transformer(nn.Module):
    def __init__(
        self, d_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None
    ):
        super(Transformer, self).__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.decoder_layer = nn.Linear(d_model, 1)
        self.device = device
        self.d_feat = d_feat

    def forward(self, src):
        # src [N, F*T] --> [N, T, F]
        src = src.reshape(len(src), self.d_feat, -1).permute(0, 2, 1)
        src = self.feature_layer(src)

        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0)  # not batch first

        mask = None

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)  # [60, 512, 8]

        # [T, N, F] --> [N, T*F]
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])  # [512, 1]

        return output.squeeze()
