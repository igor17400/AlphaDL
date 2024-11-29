import torch
import torch.nn.functional as F


def stock_mixer_loss(prediction, ground_truth, base_price, mask, batch_size, alpha):
    device = prediction.device
    all_one = torch.ones(batch_size, 1, dtype=torch.float32).to(device)
    return_ratio = (prediction - base_price) / base_price
    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
    pre_pw_dif = return_ratio @ all_one.t() - all_one @ return_ratio.t()
    gt_pw_dif = all_one @ ground_truth.t() - ground_truth @ all_one.t()
    mask_pw = mask @ mask.t()
    rank_loss = torch.mean(F.relu(pre_pw_dif * gt_pw_dif * mask_pw))
    loss = reg_loss + alpha * rank_loss
    return loss, reg_loss, rank_loss, return_ratio
