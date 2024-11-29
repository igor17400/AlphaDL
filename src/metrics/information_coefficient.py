from typing import Any, Optional
import torch
from torchmetrics import Metric


class InformationCoefficient(Metric):
    """Implementation of the Information Coefficient (IC) metric."""

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("sum_ic", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        ic = torch.corrcoef(torch.stack((preds.flatten(), targets.flatten())))[0, 1]
        self.sum_ic += ic
        self.count += 1

    def compute(self) -> torch.Tensor:
        return self.sum_ic / self.count
