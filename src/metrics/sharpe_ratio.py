from typing import Any, Optional
import torch
from torchmetrics import Metric


class SharpeRatio(Metric):
    """Implementation of the Sharpe Ratio (SR) metric."""

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, risk_free_rate: float = 0.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.risk_free_rate = risk_free_rate
        self.add_state("returns", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        returns = (preds - targets) / targets
        self.returns.append(returns)

    def compute(self) -> torch.Tensor:
        returns = torch.cat(self.returns)
        excess_returns = returns - self.risk_free_rate
        return excess_returns.mean() / excess_returns.std()
