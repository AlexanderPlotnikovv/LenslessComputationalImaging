import argparse
import time

import torch

from src.datasets.digicam import DigiCamDataset
from src.model.admm import ADMMSolver
from src.model.le_admm import LeADMM

SENSOR_SHAPE = (270, 360)
PAD_SHAPE = (540, 720)


def build_model(name, ckpt_path, device):
    if name == "admm100":
        model = ADMMSolver(SENSOR_SHAPE, PAD_SHAPE, n_iter=100, mu=1e-2, tau=2e-2, learnable=False)
    elif name == "unrolled_admm20":
        model = ADMMSolver(SENSOR_SHAPE, PAD_SHAPE, n_iter=20, mu=1e-2, tau=2e-2, learnable=True)
    elif name == "le_admm_pre_post":
        model = LeADMM(SENSOR_SHAPE, PAD_SHAPE, use_pre=True, use_post=True, n_iter=5)
    elif name == "le_admm_pre":
        model = LeADMM(SENSOR_SHAPE, PAD_SHAPE, use_pre=True, use_post=False, n_iter=5)
    elif name == "le_admm_post":
        model = LeADMM(SENSOR_SHAPE, PAD_SHAPE, use_pre=False, use_post=True, n_iter=5)
    else:
        raise ValueError(f"Unknown model: {name}")

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])

    return model.to(device).eval()


@torch.no_grad()
def measure(model, dataset, device, n_samples, n_warmup):
    times = []
    for i in range(n_warmup + n_samples):
        sample = dataset[i % len(dataset)]
        lensless = sample["lensless"].unsqueeze(0).to(device)
        psf = sample["psf"].unsqueeze(0).to(device)

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(lensless, psf)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if i >= n_warmup:
            times.append(elapsed)

    return sum(times) / len(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--n_warmup", type=int, default=5)
    parser.add_argument("--ckpt_root", default="models")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = DigiCamDataset(args.data_dir)

    models = {
        "admm100": None,
        "unrolled_admm20": f"{args.ckpt_root}/unrolled_admm20/model_best.pth",
        "le_admm_pre_post": f"{args.ckpt_root}/le_admm_pre_post/model_best.pth",
        "le_admm_pre": f"{args.ckpt_root}/le_admm_pre/model_best.pth",
        "le_admm_post": f"{args.ckpt_root}/le_admm_post/model_best.pth",
    }

    print(f"Device: {device} | samples: {args.n_samples} (warmup {args.n_warmup})\n")
    print(f"{'Model':<20} {'Time/image (ms)':>16} {'FPS':>8}")
    print("-" * 46)

    for name, ckpt in models.items():
        try:
            model = build_model(name, ckpt, device)
        except FileNotFoundError:
            print(f"{name:<20} {'checkpoint not found':>16}")
            continue
        avg = measure(model, dataset, device, args.n_samples, args.n_warmup)
        print(f"{name:<20} {avg * 1000:>16.2f} {1.0 / avg:>8.2f}")


if __name__ == "__main__":
    main()
