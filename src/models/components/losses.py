import torch
import torch.nn as nn
import torch.nn.functional as F


class StockMixerLoss(nn.Module):
    """
    Loss function for stock prediction models that combines:
    
    1. **Regression Loss (MSE)** - Measures how close the predicted returns are to the actual returns.
    2. **Pairwise Ranking Loss** - Ensures that stocks with higher actual returns are ranked above those with lower returns.

    The loss function follows the equation:
        L = L_MSE + α * sum(max(0, -(r̂ᵢ - r̂ⱼ) * (rᵢ - rⱼ)))

    where:
        - r̂ is the predicted return.
        - r is the actual return.
        - α is a weighting parameter for ranking loss.
    
    Args:
        alpha (float): Weight for the ranking loss component (default: 0.1).
    """

    def __init__(self, alpha: float = 0.1):
        """
        Initializes the StockMixerLoss module.

        Args:
            alpha (float): The weight given to the ranking loss term.
        """
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        predictions: torch.Tensor,  # Raw price predictions
        ground_truth: torch.Tensor,  # Target returns
        base_price: torch.Tensor,  # Base prices for return calculation
        mask: torch.Tensor = None,
    ) -> tuple:
        """
        Computes the combined loss for stock prediction.

        Steps:
        1. Convert predictions to returns like the original implementation.
        2. Compute the **Mean Squared Error (MSE)** between predicted and actual returns.
        3. Compute the **Pairwise Ranking Loss** to enforce correct ranking.
        4. Combine both losses using the weight `alpha`.

        Args:
            predictions (torch.Tensor): The raw price predictions.
            ground_truth (torch.Tensor): The actual observed stock returns.
            base_price (torch.Tensor): The base prices for return calculation.
            mask (torch.Tensor, optional): A binary mask indicating valid stocks (1) and invalid stocks (0).

        Returns:
            tuple: (total_loss, reg_loss, rank_loss)
                - total_loss: Combined loss (MSE + α * ranking loss).
                - reg_loss: MSE loss component.
                - rank_loss: Ranking loss component.
        """
        # Convert predictions to returns like the original implementation
        predicted_returns = torch.div(torch.sub(predictions, base_price), base_price)
        
        # Calculate losses using the converted returns
        reg_loss = F.mse_loss(predicted_returns * mask, ground_truth * mask)

        # Step 3: Compute **Pairwise Ranking Loss**
        # Computes pairwise differences:
        # pre_pw_diff(i, j) = r̂ᵢ - r̂ⱼ  (Predicted return difference)
        # gt_pw_diff(i, j) = rᵢ - rⱼ  (Actual return difference)
        pre_pw_diff = predicted_returns.unsqueeze(1) - predicted_returns.unsqueeze(0)
        gt_pw_diff = ground_truth.unsqueeze(1) - ground_truth.unsqueeze(0)

        # Apply pairwise mask only if provided
        mask_pw = mask @ mask.T if mask is not None else torch.ones_like(pre_pw_diff)

        # Compute ranking loss: max(0, -(r̂ᵢ - r̂ⱼ) * (rᵢ - rⱼ)) * mask_pw
        rank_loss = torch.mean(F.relu(-pre_pw_diff * gt_pw_diff) * mask_pw)

        # Step 4: Compute **Total Loss**
        total_loss = reg_loss + self.alpha * rank_loss

        return total_loss, reg_loss, rank_loss
