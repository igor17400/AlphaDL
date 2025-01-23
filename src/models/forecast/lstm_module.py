from typing import Any, Dict, List, Tuple, Optional
import torch
from torch import nn
from torchmetrics import MetricCollection, MeanSquaredError
from src.metrics.rank_ic import RankIC
from src.metrics.sharpe_ratio import SharpeRatio
from src.models.abstract_forecast import AbstractForecast
import pytorch_lightning as L
import torch.nn.functional as F
import wandb


class LSTMForecastModule(L.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        learning_rate: float,
        prediction_length: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, prediction_length)

        # Initialize metrics
        metrics = MetricCollection(
            {
                "mse": MeanSquaredError(),
                "rank_ic": RankIC(),
                "sharpe_ratio": SharpeRatio(),
            }
        )

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        # Log metrics
        self.train_metrics.update(y_hat, y)
        self.log("train/loss", loss, prog_bar=True)

        # Log predictions to wandb
        if batch_idx % 100 == 0:  # Log every 100 batches
            self.logger.experiment.log(
                {
                    "predictions": wandb.plot.line_series(
                        xs=range(len(y[0])),
                        ys=[
                            y[0].detach().cpu().numpy(),
                            y_hat[0].detach().cpu().numpy(),
                        ],
                        keys=["Ground Truth", "Prediction"],
                        title="Stock Price Prediction",
                    )
                }
            )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        # Log metrics
        self.val_metrics.update(y_hat, y)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        # Log metrics
        self.test_metrics.update(y_hat, y)
        self.log("test/loss", loss)
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }
