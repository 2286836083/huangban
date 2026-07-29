"""Loss functions for coordinate regression and heatmap decoding."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ALPHA_MSE_ED, HEATMAP_LOSS_ALPHA


class FoveaLoss(nn.Module):
    """Coordinate regression loss: α * MSE + (1-α) * Euclidean Distance.

    Directly optimizes for the evaluation metric (AED).
    """

    def __init__(self, alpha=ALPHA_MSE_ED):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 2] predicted (x, y) normalized
            target: [B, 2] ground truth (x, y) normalized
        Returns:
            total_loss: scalar
            loss_dict: dict for logging
        """
        mse_loss = self.mse(pred, target)
        ed_loss = torch.sqrt(((pred - target) ** 2).sum(dim=1) + 1e-8).mean()

        total = self.alpha * mse_loss + (1 - self.alpha) * ed_loss

        loss_dict = {
            "loss": total.item(),
            "mse": mse_loss.item(),
            "ed": ed_loss.item(),
        }
        return total, loss_dict


class HeatmapLoss(nn.Module):
    """热图联合损失：α * MSE(热图) + (1-α) * MSE(坐标).

    热图监督引导模型关注空间位置，坐标监督提高亚像素精度。

    Loss = α * MSE(heatmap_pred, heatmap_gt) + (1-α) * MSE(coord_pred, coord_gt)
    """

    def __init__(self, alpha=HEATMAP_LOSS_ALPHA):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred_heatmap, gt_heatmap, pred_coord, gt_coord):
        """
        Args:
            pred_heatmap: [B, 1, H, W] 预测热图
            gt_heatmap: [B, H, W] GT 热图
            pred_coord: [B, 2] 预测坐标
            gt_coord: [B, 2] GT 坐标

        Returns:
            total_loss: scalar
            loss_dict: dict
        """
        # Heatmap MSE loss
        hm_loss = self.mse(pred_heatmap, gt_heatmap.unsqueeze(1))

        # Coordinate MSE loss
        coord_loss = self.mse(pred_coord, gt_coord)

        # Combined
        total = self.alpha * hm_loss + (1 - self.alpha) * coord_loss

        loss_dict = {
            "loss": total.item(),
            "hm_loss": hm_loss.item(),
            "coord_loss": coord_loss.item(),
        }
        return total, loss_dict
