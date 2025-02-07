from typing import Dict, List, Optional, Tuple
import torch
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanSquaredError
import numpy as np
from tqdm import tqdm
import logging

from src.models.abstract_forecast import AbstractForecast
from src.models.components.encoders.stockmixer import StockMixer

logger = logging.getLogger(__name__)


class StockMixerModule(AbstractForecast):
    """
    PyTorch Lightning module for the StockMixer model.

    Attributes:
        num_stocks: Number of stocks in the dataset
        time_steps: Number of time steps to look back
        channels: Number of features per stock (OHLCV = 5)
        market: Market dimension for NoGraphMixer
        scale: Scale factor for the model
        alpha: Weight for the ranking loss
        outputs: Dictionary of outputs to collect during training
        optimizer: Optimizer to use for training
        scheduler: Learning rate scheduler
    """

    def __init__(
        self,
        num_stocks: int,
        time_steps: int,
        channels: int,
        market: int = 20,
        scale: int = 3,
        alpha: float = 0.1,
        outputs: Dict[str, List[str]] = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss: str = "mse_loss",
        learning_rate: float = 0.001,
    ):
        if outputs is None:
            outputs = {
                "train": [
                    "loss",
                    "reg_loss",
                    "rank_loss",
                    "predictions",
                    "targets",
                ],
                "val": [
                    "loss",
                    "reg_loss",
                    "rank_loss",
                    "predictions",
                    "targets",
                ],
                "test": [
                    "loss",
                    "reg_loss",
                    "rank_loss",
                    "predictions",
                    "targets",
                ],
            }

        super().__init__(
            outputs=outputs,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
        )

        self.save_hyperparameters(logger=False)

        # Initialize step outputs
        self.training_step_outputs = {}
        self.validation_step_outputs = {}
        self.test_step_outputs = {}

        self.model = StockMixer(
            num_stocks=num_stocks,
            time_steps=time_steps,
            channels=channels,
            market=market,
            scale=scale,
        )

        # Initialize metrics
        metrics = MetricCollection(
            {
                "mse": MeanSquaredError(),
            }
        )

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # Initialize optimizer with learning rate
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a single model step on a batch of data.

        Process:
        1. Model predicts closing prices for current date using historical data
        2. Convert predictions to returns using current date's actual close price
        3. Compare predicted returns with actual returns

        Args:
            batch: Tuple containing:
                - data_batch: Historical OHLCV data [batch, num_stocks, sequence_length, features]
                - price_batch: Current date's close prices [num_stocks, 1]
                - ground_truth_batch: Actual returns for current date [num_stocks, 1]
                - dates_batch: Dates information
                - current_date: Current prediction date
        """
        data_batch, price_batch, ground_truth_batch, dates_batch, current_date = batch

        # Forward pass to predict close prices for current date
        prediction = self.forward(data_batch.squeeze(0))

        # Reshape tensors to [n_stocks, 1]
        # Predicted close prices for current date
        prediction = prediction.view(-1, 1)
        # Actual returns for current date
        ground_truth_batch = ground_truth_batch.view(-1, 1)
        price_batch = price_batch.view(-1, 1)  # Current date's close prices

        # Convert predicted prices to returns
        predicted_returns = torch.div(
            torch.sub(prediction, price_batch), price_batch)

        # Calculate losses using the loss module
        loss, reg_loss, rank_loss = self.criterion(
            predicted_returns=predicted_returns,
            ground_truth=ground_truth_batch,
        )

        return {
            "loss": loss,
            "reg_loss": reg_loss,
            "rank_loss": rank_loss,
            "predictions": predicted_returns,
            "targets": ground_truth_batch,
        }

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        step_output = self.model_step(batch)

        # Log losses
        self.train_loss(step_output["loss"])
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/reg_loss", step_output["reg_loss"], on_step=False, on_epoch=True
        )
        self.log(
            "train/rank_loss", step_output["rank_loss"], on_step=False, on_epoch=True
        )

        # Update metrics
        self.train_metrics(
            step_output["predictions"],
            step_output["targets"],
        )

        # Collect step outputs for epoch end calculations
        self.training_step_outputs = self._collect_step_outputs(
            self.training_step_outputs, step_output
        )

        # Return the primary loss for backpropagation
        return step_output["loss"]

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        step_output = self.model_step(batch)

        # Log losses
        self.val_loss(step_output["loss"])
        self.log("val/loss", self.val_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("val/reg_loss",
                 step_output["reg_loss"], on_step=False, on_epoch=True)
        self.log(
            "val/rank_loss", step_output["rank_loss"], on_step=False, on_epoch=True
        )

        # Update metrics
        self.val_metrics(
            step_output["predictions"],
            step_output["targets"],
        )

        # Collect step outputs for epoch end calculations
        self.validation_step_outputs = self._collect_step_outputs(
            self.validation_step_outputs, step_output
        )

        return step_output

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        step_output = self.model_step(batch)

        # Update and log metrics
        self.test_loss(step_output["loss"])
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/reg_loss",
                 step_output["reg_loss"], on_step=False, on_epoch=True)
        self.log(
            "test/rank_loss", step_output["rank_loss"], on_step=False, on_epoch=True
        )

        # Update metrics
        self.test_metrics(
            step_output["predictions"],
            step_output["targets"],
        )

        # Collect step outputs for epoch end calculations
        self.test_step_outputs = self._collect_step_outputs(
            self.test_step_outputs, step_output
        )

        return step_output

    def on_train_epoch_end(self) -> None:
        # Log metrics
        self.log_dict(self.train_metrics.compute(),
                      on_epoch=True, prog_bar=True)

        # Clear outputs
        self.training_step_outputs = self._clear_epoch_outputs(
            self.training_step_outputs
        )

    def on_validation_epoch_end(self) -> None:
        # Log metrics
        self.log_dict(self.val_metrics.compute(), on_epoch=True, prog_bar=True)

        # Update best validation metric
        val_loss = self.val_loss.compute()
        self.val_loss_best(val_loss)
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)

        # Calculate and log IC, RIC, Precision@N, and Sharpe
        self._calculate_epoch_metrics(self.validation_step_outputs, "val")

        # Clear outputs
        self.validation_step_outputs = self._clear_epoch_outputs(
            self.validation_step_outputs
        )

    def on_test_epoch_end(self) -> None:
        # Log metrics
        self.log_dict(self.test_metrics.compute(),
                      on_epoch=True, prog_bar=True)

        # Calculate and log IC, RIC, Precision@N, and Sharpe
        self._calculate_epoch_metrics(self.test_step_outputs, "test")

        # Clear outputs
        self.test_step_outputs = self._clear_epoch_outputs(
            self.test_step_outputs)

    def _calculate_epoch_metrics(
        self, step_outputs: Dict[str, List[torch.Tensor]], stage: str
    ) -> None:
        """Calculate epoch-level metrics including IC, RIC, Precision@N, and Sharpe ratio."""
        predictions = self._gather_step_outputs(step_outputs, "predictions")
        targets = self._gather_step_outputs(step_outputs, "targets")

        # Calculate and log each metric
        ic = self._calculate_ic(predictions, targets)
        ric = self._calculate_ric(predictions, targets)
        precision = self._calculate_precision_at_n(predictions, targets, n=10)
        sharpe = self._calculate_sharpe_ratio(predictions, targets)

        # Log metrics
        self.log(f"{stage}/IC", ic, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/RIC", ric, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/Precision@10", precision,
                 on_epoch=True, prog_bar=True)
        self.log(f"{stage}/Sharpe", sharpe, on_epoch=True, prog_bar=True)

    def _calculate_ic(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculate Information Coefficient (IC) using Pearson correlation for the current timestamp.
        """
        # Ensure we're working with the right shape
        predictions = predictions.squeeze()
        targets = targets.squeeze()

        n = len(predictions)
        if n < 2:  # Need at least 2 points for correlation
            return 0.0

        # Standardize the values (zero mean, unit variance)
        pred_std = (predictions - predictions.mean()) / predictions.std()
        target_std = (targets - targets.mean()) / targets.std()

        # Calculate Pearson correlation
        pearson_corr = (pred_std * target_std).mean()

        return pearson_corr.item()

    def _calculate_ric(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculate Rank Information Coefficient (RIC) using Spearman rank correlation.
        RIC measures the correlation between the rankings of predicted and actual returns.
        """
        # Ensure we're working with the right shape
        predictions = predictions.squeeze()
        targets = targets.squeeze()

        # Convert tensors to numpy for ranking
        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()

        # Calculate rankings
        pred_ranks = np.argsort(np.argsort(pred_np))
        target_ranks = np.argsort(np.argsort(target_np))

        n = len(predictions)
        if n < 2:  # Need at least 2 points for correlation
            return 0.0

        # Calculate Spearman correlation
        # Formula: 1 - (6 * sum(d²) / (n * (n² - 1))), where d is rank difference
        d = pred_ranks - target_ranks
        spearman_corr = 1 - (6 * np.sum(d**2)) / (n * (n**2 - 1))

        return float(spearman_corr)

    def _calculate_precision_at_n(self, predictions: torch.Tensor, targets: torch.Tensor, n: int = 10) -> float:
        """
        Calculate Precision@N for the current timestamp.
        Precision@N measures how many of our predicted positive returns actually resulted in positive returns.
        For example, if we predicted 10 positive returns and 4 of them actually were positive, Precision@10 = 40%.
        """
        # Ensure we're working with the right shape
        predictions = predictions.squeeze()
        targets = targets.squeeze()

        if len(predictions) < n:
            return 0.0

        # Get indices of top N predictions based on predicted returns
        _, top_indices = torch.topk(predictions, n)

        # Get predictions and actual returns for top N predictions
        top_predictions = predictions[top_indices]
        top_targets = targets[top_indices]

        # Count where we predicted positive returns (prediction > 0)
        # and the actual returns were also positive (target > 0)
        true_positives = ((top_predictions > 0) & (
            top_targets > 0)).float().sum()
        # Count total positive predictions
        predicted_positives = (top_predictions > 0).float().sum()

        # Calculate precision
        # If we had no positive predictions, return 0 to avoid division by zero
        precision = (true_positives / predicted_positives.clamp(min=1)).item()

        return precision

    def _calculate_sharpe_ratio(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        freq: str = 'day',
        annual_rf_rate: float = None
    ) -> float:
        """
        Calculate Sharpe ratio using top 5 predictions for the current timestamp.
        SR = (R - Rf) / σ, where:
        - R is the portfolio return (annualized)
        - Rf is the risk-free rate (annualized)
        - σ is the annualized standard deviation of returns

        Args:
            predictions: Predicted returns
            targets: Actual returns
            freq: Data frequency ('1min', '5min', '15min', 'hour', 'day')
            annual_rf_rate: Annual risk-free rate (e.g., 0.045 for 4.5%)
                          If None, assumes 0 (excess returns)

        Annualization factors (assuming 6.5 trading hours):
        - 1-minute: √(252 * 6.5 * 60) ≈ √98,280 ≈ 313.5
        - 5-minute: √(252 * 6.5 * 12) ≈ √19,656 ≈ 140.2
        - 15-minute: √(252 * 6.5 * 4) ≈ √6,552 ≈ 80.9
        - Hours: √(252 * 6.5) ≈ √1,638 ≈ 40.5
        - Days: √252 ≈ 15.87
        """
        # Ensure we're working with the right shape
        predictions = predictions.squeeze()
        targets = targets.squeeze()

        if len(predictions) < 5:
            return 0.0

        # Define annualization factors for different frequencies
        trading_days = 252
        trading_hours = 6.5
        annualization_factors = {
            '1min': np.sqrt(trading_days * trading_hours * 60),    # ~313.5
            '5min': np.sqrt(trading_days * trading_hours * 12),    # ~140.2
            '15min': np.sqrt(trading_days * trading_hours * 4),    # ~80.9
            'hour': np.sqrt(trading_days * trading_hours),         # ~40.5
            'day': np.sqrt(trading_days),                         # ~15.87
        }

        if freq not in annualization_factors:
            raise ValueError(
                f"Unsupported frequency: {freq}. Use '1min', '5min', '15min', 'hour', or 'day'")

        annualization_factor = annualization_factors[freq]

        # Get top 5 predictions and their actual returns
        _, top_indices = torch.topk(predictions, 5)
        top_returns = targets[top_indices]

        # Calculate mean return and standard deviation
        mean_return = top_returns.mean().item()
        std_return = top_returns.std().item()

        # Annualize returns and volatility
        annual_return = mean_return * annualization_factor
        annual_std = std_return * annualization_factor

        # If no risk-free rate provided, calculate excess returns (Rf = 0)
        rf_rate = 0.0 if annual_rf_rate is None else annual_rf_rate

        # Calculate Sharpe ratio
        # If std is 0, return 0 to avoid division by zero
        return (annual_return - rf_rate) / annual_std if annual_std != 0 else 0.0

    def _collect_step_outputs(self, outputs_dict: Dict[str, List], step_output: Dict) -> Dict[str, List]:
        """Collect step outputs for epoch-end processing."""
        for key in step_output:
            if key not in outputs_dict:
                outputs_dict[key] = []
            outputs_dict[key].append(step_output[key].detach())
        return outputs_dict

    def _clear_epoch_outputs(self, outputs_dict: Dict[str, List]) -> Dict[str, List]:
        """Clear the outputs dictionary after epoch end."""
        return {}
