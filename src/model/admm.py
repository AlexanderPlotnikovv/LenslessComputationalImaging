import torch
import torch.nn as nn
import torch.nn.functional as F


def finite_diff(x):
    return torch.stack([
        torch.roll(x, -1, -2) - x,
        torch.roll(x, -1, -1) - x,
    ], dim=-1)


def finite_diff_adj(d):
    dx, dy = d[..., 0], d[..., 1]
    return (torch.roll(dx, 1, -2) - dx) + (torch.roll(dy, 1, -1) - dy)


def soft_threshold(x, t):
    return torch.sign(x) * F.relu(x.abs() - t)


def psf_to_otf(psf, pad_shape):
    H, W = pad_shape
    ph = H - psf.shape[-2]
    pw = W - psf.shape[-1]
    psf_pad = F.pad(psf, (0, pw, 0, ph))
    psf_pad = torch.roll(psf_pad, (-psf.shape[-2] // 2, -psf.shape[-1] // 2), (-2, -1))
    return torch.fft.rfft2(psf_pad)


class ADMMSolver(nn.Module):
    def __init__(self, sensor_shape, pad_shape, n_iter=100, mu=1e-4, tau=2e-4, learnable=False):
        super().__init__()
        self.n_iter = n_iter
        self.sensor_shape = tuple(sensor_shape)
        self.pad_shape = tuple(pad_shape)

        if learnable:
            self.log_mu = nn.Parameter(torch.full((n_iter,), torch.log(torch.tensor(mu))))
            self.log_tau = nn.Parameter(torch.full((n_iter,), torch.log(torch.tensor(tau))))
        else:
            self.register_buffer("log_mu", torch.full((n_iter,), torch.log(torch.tensor(mu))))
            self.register_buffer("log_tau", torch.full((n_iter,), torch.log(torch.tensor(tau))))

    def _pad(self, y):
        H_pad, W_pad = self.pad_shape
        H_s, W_s = self.sensor_shape
        ph = (H_pad - H_s) // 2
        pw = (W_pad - W_s) // 2
        return F.pad(y, (pw, W_pad - W_s - pw, ph, H_pad - H_s - ph))

    def _crop(self, x):
        H_pad, W_pad = self.pad_shape
        H_s, W_s = self.sensor_shape
        ph = (H_pad - H_s) // 2
        pw = (W_pad - W_s) // 2
        return x[..., ph: ph + H_s, pw: pw + W_s]

    def _x_update(self, y, psf_otf, rhs, mu):
        Hty = torch.fft.irfft2(psf_otf.conj() * torch.fft.rfft2(self._pad(y)), s=self.pad_shape)
        RHS_f = torch.fft.rfft2(Hty + rhs)
        denom = psf_otf.abs() ** 2 + mu
        return torch.fft.irfft2(RHS_f / denom, s=self.pad_shape)

    def forward(self, lensless, psf, **batch):
        B, C, H_s, W_s = lensless.shape
        H_pad, W_pad = self.pad_shape

        psf_otf = psf_to_otf(psf, self.pad_shape)

        x = torch.zeros(B, C, H_pad, W_pad, device=lensless.device, dtype=lensless.dtype)
        z = torch.zeros(B, C, H_pad, W_pad, 2, device=lensless.device, dtype=lensless.dtype)
        u = torch.zeros_like(z)

        for k in range(self.n_iter):
            mu_k = self.log_mu[k].exp()
            tau_k = self.log_tau[k].exp()

            x = self._x_update(lensless, psf_otf, mu_k * finite_diff_adj(z - u), mu_k)
            z = soft_threshold(finite_diff(x) + u, tau_k / mu_k)
            u = u + finite_diff(x) - z

        return {"reconstruction": self._crop(x).clamp(0, 1)}
