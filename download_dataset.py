import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def save_split(hf_split, split_dir: Path, masks_dir: Path):
    (split_dir / "lensless").mkdir(parents=True, exist_ok=True)
    (split_dir / "lensed").mkdir(parents=True, exist_ok=True)
    (split_dir / "masks").mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(tqdm(hf_split, desc=split_dir.name)):
        image_id = f"{i:05d}"
        mask_label = sample["mask_label"]

        sample["lensless"].save(split_dir / "lensless" / f"{image_id}.png")
        sample["lensed"].save(split_dir / "lensed" / f"{image_id}.png")

        src_mask = masks_dir / f"mask_{mask_label}.npy"
        dst_mask = split_dir / "masks" / f"{image_id}.npy"
        dst_mask.write_bytes(src_mask.read_bytes())


def download_masks(masks_dir: Path):
    masks_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading 100 PSF masks...")
    for i in tqdm(range(100)):
        dst = masks_dir / f"mask_{i}.npy"
        if dst.exists():
            continue
        path = hf_hub_download(
            repo_id="bezzam/DigiCam-Mirflickr-MultiMask-10K",
            filename=f"masks/mask_{i}.npy",
            repo_type="dataset",
        )
        import shutil
        shutil.copy(path, dst)


def precompute_psfs(split_dir, output_shape=(270, 360)):
    from src.utils.psf_utils import simulate_psf
    psf_dir = split_dir / "psfs"
    psf_dir.mkdir(exist_ok=True)
    for mask_path in tqdm(sorted((split_dir / "masks").glob("*.npy")), desc=f"PSFs {split_dir.name}"):
        psf_path = psf_dir / mask_path.name
        if psf_path.exists():
            continue
        np.save(psf_path, simulate_psf(np.load(mask_path), output_shape).numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data", type=str)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    masks_dir = data_dir / "masks_raw"

    download_masks(masks_dir)

    print("Loading dataset from HuggingFace...")
    dataset = load_dataset(
        "bezzam/DigiCam-Mirflickr-MultiMask-10K",
        cache_dir=str(data_dir / ".cache"),
    )

    save_split(dataset["train"], data_dir / "train", masks_dir)
    save_split(dataset["test"], data_dir / "test", masks_dir)

    precompute_psfs(data_dir / "train")
    precompute_psfs(data_dir / "test")

    print(f"\nDone! Dataset saved to {data_dir}")
    print(f"  train: {len(dataset['train'])} samples")
    print(f"  test:  {len(dataset['test'])} samples")


if __name__ == "__main__":
    main()
