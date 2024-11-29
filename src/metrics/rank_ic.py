from typing import Any, Optional
import torch
from torchmetrics import Metric


class RankIC(Metric):
    """Implementation of the Rank Information Coefficient (RIC) metric."""

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("sum_ric", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds_rank = preds.argsort().argsort()
        targets_rank = targets.argsort().argsort()
        ric = torch.corrcoef(torch.stack((preds_rank.float(), targets_rank.float())))[
            0, 1
        ]
        self.sum_ric += ric
        self.count += 1

    def compute(self) -> torch.Tensor:
        return self.sum_ric / self.count
