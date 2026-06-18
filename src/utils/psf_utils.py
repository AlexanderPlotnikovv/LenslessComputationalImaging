import numpy as np
import torch
import torch.nn.functional as F

SCENE2MASK = 0.30
MASK2SENSOR = 0.002
WAVELENGTHS = {
    0: 640e-9,
    1: 550e-9,
    2: 460e-9,
}
SUBPIXEL_SIZE = (0.06e-3, 0.18e-3)


def simulate_psf(
        mask: np.ndarray,
        output_shape: tuple,
) -> torch.Tensor:
    from waveprop.slm import get_intensity_psf
    from waveprop.rs import angular_spectrum

    psf_channels = []
    for c in range(3):
        wl = WAVELENGTHS[c]
        mask_c = mask[c]
        psf_c = get_intensity_psf(
            mask=mask_c,
            wv=wl,
            d1=SUBPIXEL_SIZE,
            dz=MASK2SENSOR,
        )
        psf_channels.append(psf_c)

    psf = np.stack(psf_channels, axis=0)
    psf = torch.from_numpy(psf.astype(np.float32))

    psf = F.interpolate(
        psf.unsqueeze(0),
        size=output_shape,
        mode='bilinear',
        align_corners=False,
    ).squeeze(0)

    psf = psf / psf.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    return psf
