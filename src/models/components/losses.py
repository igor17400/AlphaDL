import torch
import torch.nn as nn
import torch.nn.functional as F


class StockMixerLoss(nn.Module):
    """Loss function for StockMixer model combining regression and ranking losses.

    Args:
        alpha (float): Weight for the ranking loss component
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        base_price: torch.Tensor,
    ) -> tuple:
        """
        Calculate combined loss for stock prediction.

        Args:
            prediction: Model predictions [batch_size, 1]
            ground_truth: Actual returns [batch_size, 1]
            base_price: Base prices [batch_size, 1]

        Returns:
            tuple: (total_loss, reg_loss, rank_loss, return_ratio)
        """
        batch_size = prediction.shape[0]
        device = prediction.device

        # Calculate return ratio
        all_one = torch.ones(batch_size, 1, dtype=torch.float32).to(device)
        return_ratio = torch.div(torch.sub(prediction, base_price), base_price)

        # Regression loss (MSE)
        reg_loss = F.mse_loss(return_ratio, ground_truth)

        # Ranking loss
        pre_pw_dif = torch.sub(return_ratio @ all_one.t(), all_one @ return_ratio.t())
        gt_pw_dif = torch.sub(ground_truth @ all_one.t(), all_one @ ground_truth.t())
        rank_loss = torch.mean(F.relu(pre_pw_dif * gt_pw_dif))

        # Combined loss
        total_loss = reg_loss + self.alpha * rank_loss

        return total_loss, reg_loss, rank_loss, return_ratio
