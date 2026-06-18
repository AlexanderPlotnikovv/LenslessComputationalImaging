import torch
from torch import nn
import lpips


class CombinedLoss(nn.Module):
    def __init__(self, lpips_weight=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lpips = lpips.LPIPS(net="vgg")
        self.lpips_weight = lpips_weight

    def forward(self, reconstruction, lensed, **batch):
        mse_loss = self.mse(reconstruction, lensed)
        lpips_loss = self.lpips(
            reconstruction * 2 - 1,
            lensed * 2 - 1,
        ).mean()
        return {
            "loss": mse_loss + self.lpips_weight * lpips_loss,
            "mse_loss": mse_loss,
            "lpips_loss": lpips_loss,
        }
