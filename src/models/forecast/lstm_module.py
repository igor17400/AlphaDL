from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanSquaredError
from src.metrics.information_coefficient import InformationCoefficient
from src.metrics.rank_ic import RankIC
from src.metrics.sharpe_ratio import SharpeRatio
from src.models.abstract_forecast import AbstractForecast
from src.models.components.lstm import LSTM


class LSTMForecastModule(AbstractForecast):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        learning_rate: float,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Define LSTM model
        self.lstm = LSTM(
            input_size=self.hparams.input_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
            dropout=self.hparams.dropout,
        )
        self.fc = nn.Linear(self.hparams.hidden_size, 1)

        # Define metrics
        self.train_metrics = MetricCollection(
            {
                "mse": MeanSquaredError(),
                "ic": InformationCoefficient(),
                "ric": RankIC(),
                "sr": SharpeRatio(),
            },
            prefix="train/",
        )

        self.val_metrics = self.train_metrics.clone(prefix="val/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train/loss", loss)

        # Update metrics
        self.train_metrics(y_hat, y)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("val/loss", loss)

        # Update metrics
        self.val_metrics(y_hat, y)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("test/loss", loss)

        # Update metrics
        self.test_metrics(y_hat, y)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return self.hparams.optimizer(self.parameters(), lr=self.hparams.learning_rate)
