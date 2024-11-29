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
        loss: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        learning_rate: float,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ) -> None:
        # Define the outputs attribute
        outputs = {
            "train": ["preds", "targets"],
            "val": ["preds", "targets"],
            "test": ["preds", "targets"],
        }

        super().__init__(
            outputs=outputs,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        self.save_hyperparameters()

        # Define LSTM model
        self.lstm = LSTM(
            input_size=self.hparams.input_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
            dropout=self.hparams.dropout,
        )
        self.fc = nn.Linear(self.hparams.hidden_size, 1)

        # Initialize loss function
        self.criterion = self._get_loss(self.hparams.loss)

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

        # Initialize step outputs
        self.training_step_outputs = {key: [] for key in outputs["train"]}
        self.val_step_outputs = {key: [] for key in outputs["val"]}
        self.test_step_outputs = {key: [] for key in outputs["test"]}

    def forward(self, x):
        return self.lstm(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # print("--- y ---")
        # print(y.size())
        # print(y)
        # print("--- y_hat ---")
        # print(y_hat.size())
        # print(y_hat)
        loss = self.criterion(y_hat, y)
        # print("--- loss ---")
        # print(loss)
        self.log("train/loss", loss)

        # Update metrics
        self.train_metrics(y_hat, y)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)

        # Collect step outputs for metric computation
        self.training_step_outputs = self._collect_step_outputs(
            outputs_dict=self.training_step_outputs,
            local_vars={"preds": y_hat, "targets": y}
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val/loss", loss)

        # Update metrics
        self.val_metrics(y_hat, y)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)

        # Collect step outputs for metric computation
        self.val_step_outputs = self._collect_step_outputs(
            outputs_dict=self.val_step_outputs,
            local_vars={"preds": y_hat, "targets": y}
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("test/loss", loss)

        # Update metrics
        self.test_metrics(y_hat, y)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)

        # Collect step outputs for metric computation
        self.test_step_outputs = self._collect_step_outputs(
            outputs_dict=self.test_step_outputs,
            local_vars={"preds": y_hat, "targets": y}
        )

    def configure_optimizers(self):
        return self.hparams.optimizer(self.parameters(), lr=self.hparams.learning_rate)

    def on_train_epoch_end(self) -> None:
        # Gather predictions and targets
        preds = self._gather_step_outputs(self.training_step_outputs, "preds")
        targets = self._gather_step_outputs(self.training_step_outputs, "targets")

        # Update and log metrics
        self.train_metrics(preds, targets)
        self.log_dict(
            self.train_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        # Clear memory for the next epoch
        self.training_step_outputs = self._clear_epoch_outputs(
            self.training_step_outputs
        )

    def on_validation_epoch_end(self) -> None:
        # Gather predictions and targets
        preds = self._gather_step_outputs(self.val_step_outputs, "preds")
        targets = self._gather_step_outputs(self.val_step_outputs, "targets")

        # Update and log metrics
        self.val_metrics(preds, targets)
        self.log_dict(
            self.val_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        # Clear memory for the next epoch
        self.val_step_outputs = self._clear_epoch_outputs(self.val_step_outputs)

    def on_test_epoch_end(self) -> None:
        # Gather predictions and targets
        preds = self._gather_step_outputs(self.test_step_outputs, "preds")
        targets = self._gather_step_outputs(self.test_step_outputs, "targets")

        # Update and log metrics
        self.test_metrics(preds, targets)
        self.log_dict(
            self.test_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        # Clear memory for the next epoch
        self.test_step_outputs = self._clear_epoch_outputs(self.test_step_outputs)

    def _collect_step_outputs(self, outputs_dict, local_vars):
        # Collect step outputs for metric computation
        for key, value in local_vars.items():
            if key not in outputs_dict:
                outputs_dict[key] = []
            outputs_dict[key].append(value)
        return outputs_dict
