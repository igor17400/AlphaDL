from typing import Dict, List, Optional, Tuple
import torch
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanSquaredError
import numpy as np

from src.models.abstract_forecast import AbstractForecast
from src.models.components.encoders.stockmixer import StockMixer


class StockMixerModule(AbstractForecast):
    """
    PyTorch Lightning module for the StockMixer model.

    Attributes:
        stocks: Number of stocks in the dataset
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
        stocks: int,
        time_steps: int,
        channels: int,
        market: int = 20,
        scale: int = 3,
        alpha: float = 0.1,
        outputs: Dict[str, List[str]] = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss: str = "mse_loss",
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
            stocks=stocks,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Perform a single model step on a batch of data."""
        data_batch, price_batch, ground_truth_batch, dates_batch = batch
        
        # Assuming that the batch size is equal to 1
        data_batch = data_batch.squeeze(0)
        price_batch = price_batch.squeeze(0)
        ground_truth_batch = ground_truth_batch.squeeze(0)
        
        # Forward pass
        prediction = self.forward(data_batch)
        
        # Calculate losses using the loss module
        loss, reg_loss, rank_loss, return_ratio = self.criterion(
            prediction=prediction,
            ground_truth=ground_truth_batch,
            base_price=price_batch,
        )
        
        return {
            "loss": loss,
            "reg_loss": reg_loss,
            "rank_loss": rank_loss,
            "predictions": return_ratio,
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

        return step_output

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        step_output = self.model_step(batch)

        # Log losses
        self.val_loss(step_output["loss"])
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/reg_loss", step_output["reg_loss"], on_step=False, on_epoch=True)
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
        self.log("test/reg_loss", step_output["reg_loss"], on_step=False, on_epoch=True)
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
        self.log_dict(self.train_metrics.compute(), on_epoch=True, prog_bar=True)

        # Calculate and log IC
        self._calculate_epoch_metrics(self.training_step_outputs, "train")

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

        # Calculate and log IC
        self._calculate_epoch_metrics(self.validation_step_outputs, "val")

        # Clear outputs
        self.validation_step_outputs = self._clear_epoch_outputs(
            self.validation_step_outputs
        )

    def on_test_epoch_end(self) -> None:
        # Log metrics
        self.log_dict(self.test_metrics.compute(), on_epoch=True, prog_bar=True)

        # Calculate and log IC
        self._calculate_epoch_metrics(self.test_step_outputs, "test")

        # Clear outputs
        self.test_step_outputs = self._clear_epoch_outputs(self.test_step_outputs)

    def _calculate_epoch_metrics(
        self, step_outputs: Dict[str, List[torch.Tensor]], stage: str
    ) -> None:
        """Calculate epoch-level metrics including IC, RIC, Precision@N, and Sharpe ratio."""
        predictions = self._gather_step_outputs(step_outputs, "predictions")
        targets = self._gather_step_outputs(step_outputs, "targets")
        
        # Calculate metrics without mask
        ic = self._calculate_ic(predictions, targets)
        self.log(f"{stage}/IC", ic, on_epoch=True, prog_bar=True)
        
        ric = self._calculate_ric(predictions, targets)
        self.log(f"{stage}/RIC", ric, on_epoch=True, prog_bar=True)
        
        prec_10 = self._calculate_precision_at_n(predictions, targets, n=10)
        self.log(f"{stage}/Precision@10", prec_10, on_epoch=True, prog_bar=True)
        
        sharpe = self._calculate_sharpe_ratio(predictions, targets)
        self.log(f"{stage}/Sharpe", sharpe, on_epoch=True, prog_bar=True)

    def _calculate_ic(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculate Information Coefficient (Spearman rank correlation) for the current timestamp.
        """
        # Ensure we're working with the right shape
        predictions = predictions.squeeze()
        targets = targets.squeeze()
        
        # Convert to ranks
        pred_ranks = predictions.argsort().argsort().float()
        target_ranks = targets.argsort().argsort().float()
        
        # Calculate Spearman correlation
        n = len(predictions)
        if n < 2:  # Need at least 2 points for correlation
            return 0.0
        
        # Standardize the ranks
        pred_ranks = (pred_ranks - pred_ranks.mean()) / pred_ranks.std()
        target_ranks = (target_ranks - target_ranks.mean()) / target_ranks.std()
        
        # Calculate correlation
        spearman_corr = (pred_ranks * target_ranks).mean()
        
        return spearman_corr.item()

    def _calculate_ric(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculate Rank Information Coefficient (RIC).
        RIC is IC normalized by its standard deviation: RIC = mean(IC) / std(IC)
        """
        # Ensure we're working with the right shape
        predictions = predictions.squeeze()
        targets = targets.squeeze()
        
        # Calculate IC for each stock
        ic_values = []
        for i in range(len(predictions)):
            if i + 1 < len(predictions):  # ensure we have at least 2 points
                pred_window = predictions[i:i+2]  # use sliding window of 2
                target_window = targets[i:i+2]
                ic = self._calculate_ic(pred_window, target_window)
                ic_values.append(ic)
        
        ic_values = np.array(ic_values)
        
        # Calculate RIC as mean(IC)/std(IC)
        if len(ic_values) > 1 and np.std(ic_values) != 0:
            return np.mean(ic_values) / np.std(ic_values)
        return 0.0

    def _calculate_precision_at_n(self, predictions: torch.Tensor, targets: torch.Tensor, n: int = 10) -> float:
        """
        Calculate Precision@N for the current timestamp.
        Precision@N is the percentage of positive returns in the top N predictions.
        For example, if 4 out of top 10 predictions have positive returns, Precision@10 = 40%.
        """
        # Ensure we're working with the right shape
        predictions = predictions.squeeze()
        targets = targets.squeeze()
        
        if len(predictions) < n:
            return 0.0
        
        # Get indices of top N predictions based on predicted returns
        _, top_indices = torch.topk(predictions, n)
        
        # Get actual returns for top N predictions
        top_targets = targets[top_indices]
        
        # Calculate percentage of positive returns in top N predictions
        # (count of returns >= 0) / N * 100
        precision = (top_targets >= 0).float().mean().item() * 100
        
        return precision

    def _calculate_sharpe_ratio(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculate Sharpe ratio using top 5 predictions for the current timestamp.
        SR = (R - Rf) / σ, where:
        - R is the portfolio return
        - Rf is the risk-free rate (assumed 0)
        - σ is the standard deviation of returns
        """
        # Ensure we're working with the right shape
        predictions = predictions.squeeze()
        targets = targets.squeeze()
        
        if len(predictions) < 5:
            return 0.0
        
        # Get top 5 predictions and their actual returns
        _, top_indices = torch.topk(predictions, 5)
        top_returns = targets[top_indices]
        
        # Calculate mean return and standard deviation
        mean_return = top_returns.mean().item()
        std_return = top_returns.std().item()
        
        # Calculate Sharpe ratio (assuming Rf = 0)
        # If std is 0, return 0 to avoid division by zero
        return mean_return / std_return if std_return != 0 else 0.0

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
