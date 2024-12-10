from typing import Any, Dict, List, Tuple, Optional
import torch
from torch import nn
from torchmetrics import MetricCollection, MeanSquaredError
from src.metrics.rank_ic import RankIC
from src.metrics.sharpe_ratio import SharpeRatio
from src.models.abstract_forecast import AbstractForecast


class LSTMForecastModule(AbstractForecast):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        outputs: Dict[str, List[str]],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss: str = "mse_loss",
    ) -> None:
        super().__init__(
            outputs=outputs, optimizer=optimizer, scheduler=scheduler, loss=loss
        )

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

        # Define metrics
        self.metrics = MetricCollection(
            {
                "mse": MeanSquaredError(),
                "rank_ic": RankIC(),
                "sharpe_ratio": SharpeRatio(),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("---")
        print(x)
        print("---")
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        return loss, preds, y

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.metrics.update(preds, targets)

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.metrics.compute(), prog_bar=True)
        self.metrics.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        return super().configure_optimizers()
