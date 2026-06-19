import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from torchmetrics.image import peak_signal_noise_ratio, structural_similarity_index_measure
except ImportError:
    from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchvision import transforms
from tqdm import tqdm
import lpips


def load_image(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size[1], size[0]))
    return transforms.ToTensor()(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)

    lpips_fn = lpips.LPIPS(net="vgg")

    psnr_vals, ssim_vals, mse_vals, lpips_vals = [], [], [], []

    pred_paths = sorted(pred_dir.glob("*.png"))
    assert len(pred_paths) > 0, f"No predictions found in {pred_dir}"

    for pred_path in tqdm(pred_paths):
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            continue

        gt = load_image(gt_path).unsqueeze(0)
        pred = load_image(pred_path).unsqueeze(0)

        psnr_vals.append(peak_signal_noise_ratio(pred, gt, data_range=1.0).item())
        ssim_vals.append(structural_similarity_index_measure(pred, gt, data_range=1.0).item())
        mse_vals.append(torch.nn.functional.mse_loss(pred, gt).item())
        lpips_vals.append(lpips_fn(pred * 2 - 1, gt * 2 - 1).mean().item())

    print(f"\nMetrics on {len(psnr_vals)} samples:")
    print(f"  PSNR:  {np.mean(psnr_vals):.4f}")
    print(f"  SSIM:  {np.mean(ssim_vals):.4f}")
    print(f"  MSE:   {np.mean(mse_vals):.6f}")
    print(f"  LPIPS: {np.mean(lpips_vals):.4f}")


if __name__ == "__main__":
    main()
