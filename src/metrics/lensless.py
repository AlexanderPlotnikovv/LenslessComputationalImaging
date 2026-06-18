import torch
import lpips
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

from src.metrics.base_metric import BaseMetric


class PSNRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, reconstruction, lensed, **batch):
        return peak_signal_noise_ratio(reconstruction, lensed, data_range=1.0).item()


class SSIMMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, reconstruction, lensed, **batch):
        return structural_similarity_index_measure(reconstruction, lensed, data_range=1.0).item()


class MSEMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, reconstruction, lensed, **batch):
        return torch.nn.functional.mse_loss(reconstruction, lensed).item()


class LPIPSMetric(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.lpips = lpips.LPIPS(net="vgg").to(device)

    @torch.no_grad()
    def __call__(self, reconstruction, lensed, **batch):
        return self.lpips(
            reconstruction.to(self.device) * 2 - 1,
            lensed.to(self.device) * 2 - 1,
        ).mean().item()
