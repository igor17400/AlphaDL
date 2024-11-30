from typing import Optional
import torch
from pytorch_forecasting.models import LSTM
from torchmetrics.regression import MeanSquaredError
from src.metrics.rank_ic import RankIC
from src.metrics.sharpe_ratio import SharpeRatio


class LSTMForecastModule:
    def __init__(self, dataset, **kwargs):
        self.model = LSTM.from_dataset(
            dataset,
            learning_rate=kwargs.get("learning_rate", 0.03),
            hidden_size=kwargs.get("hidden_size", 64),
            dropout=kwargs.get("dropout", 0.1),
            loss=MeanSquaredError(),
            log_interval=10,
            log_val_interval=1,
        )
        # Instantiate metrics
        self.mse = MeanSquaredError()
        self.rank_ic = RankIC()
        self.sharpe_ratio = SharpeRatio(risk_free_rate=kwargs.get("risk_free_rate", 0.0))

    def fit(self, train_dataloader, val_dataloader, max_epochs=20):
        self.model.fit(
            train_dataloader,
            val_dataloader=val_dataloader,
            max_epochs=max_epochs,
            gpus=1 if torch.cuda.is_available() else 0,
        )

    def predict(self, dataloader):
        return self.model.predict(dataloader)

    def evaluate(self, dataloader):
        self.mse.reset()
        self.rank_ic.reset()
        self.sharpe_ratio.reset()

        for x, y in dataloader:
            preds = self.predict(x)
            self.mse.update(preds, y)
            self.rank_ic.update(preds, y)
            self.sharpe_ratio.update(preds, y)

        mse_value = self.mse.compute()
        rank_ic_value = self.rank_ic.compute()
        sharpe_ratio_value = self.sharpe_ratio.compute()

        print(f"Validation MSE: {mse_value}")
        print(f"Validation RankIC: {rank_ic_value}")
        print(f"Validation Sharpe Ratio: {sharpe_ratio_value}")

        return {
            "mse": mse_value,
            "rank_ic": rank_ic_value,
            "sharpe_ratio": sharpe_ratio_value,
        }
