import torch.nn as nn
from src.model.admm import ADMMSolver
from src.model.drunet import DRUNet


class LeADMM(nn.Module):
    def __init__(self, sensor_shape, pad_shape, use_pre=True, use_post=True, n_iter=5, base_ch=64):
        super().__init__()
        self.use_pre = use_pre
        self.use_post = use_post
        if use_pre:
            self.pre = DRUNet(base_ch=base_ch)
        self.admm = ADMMSolver(sensor_shape=sensor_shape, pad_shape=pad_shape, n_iter=n_iter, learnable=True)
        if use_post:
            self.post = DRUNet(base_ch=base_ch)

    def forward(self, lensless, psf, **batch):
        x = self.pre(lensless) if self.use_pre else lensless
        out = self.admm(x, psf)
        recon = out["reconstruction"]
        recon = self.post(recon) if self.use_post else recon
        return {"reconstruction": recon}
