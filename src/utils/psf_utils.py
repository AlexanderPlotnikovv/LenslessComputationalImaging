import numpy as np
import torch
import torch.nn.functional as F

MASK2SENSOR = 0.002
WAVELENGTHS = [640e-9, 550e-9, 460e-9]
SUBPIXEL_H = 0.06e-3


def blas_transfer_function(shape, pixel_size, distance, wavelength, device):
    H, W = shape

    fy = torch.fft.fftfreq(H, d=pixel_size, device=device)
    fx = torch.fft.fftfreq(W, d=pixel_size, device=device)

    FY, FX = torch.meshgrid(fy, fx, indexing="ij")

    k = 2 * torch.pi / wavelength
    factor = 1 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
    mask = factor > 0

    phase = torch.zeros(H, W, dtype=torch.complex64, device=device)
    phase[mask] = torch.exp(1j * k * distance * torch.sqrt(factor[mask]))
    return phase


def simulate_psf(mask, output_shape, device=torch.device("cpu")):
    mask_np = mask.astype(np.float32)

    if mask_np.ndim == 2:
        mask_np = np.stack([mask_np] * 3, axis=0)

    mask_t = torch.from_numpy(mask_np).to(device)

    C, H_m, W_m = mask_t.shape
    psf_channels = []
    for c in range(C):
        field = mask_t[c].to(torch.complex64)
        field_f = torch.fft.fft2(field)

        H_tf = blas_transfer_function((H_m, W_m), SUBPIXEL_H, MASK2SENSOR, WAVELENGTHS[c], device)
        field_sensor = torch.fft.ifft2(field_f * H_tf)
        psf_channels.append(field_sensor.abs() ** 2)

    psf = torch.stack(psf_channels, dim=0)
    psf = F.interpolate(psf.unsqueeze(0), size=output_shape, mode="bilinear", align_corners=False).squeeze(0)
    psf = psf / psf.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    return psf.float()
