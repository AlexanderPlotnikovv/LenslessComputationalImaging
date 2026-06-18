import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

logger = logging.getLogger(__name__)


class DigiCamDataset(BaseDataset):
    def __init__(
            self,
            data_dir: str,
            image_size: tuple = (270, 360),
            name: str = "train",
            *args,
            **kwargs,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size

        self.to_tensor = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

        index = self._create_index()
        super().__init__(index, *args, **kwargs)

    def _create_index(self):
        lensless_dir = self.data_dir / "lensless"
        lensed_dir = self.data_dir / "lensed"
        psfs_dir = self.data_dir / "psfs"

        assert lensless_dir.exists(), f"lensless dir not found: {lensless_dir}"
        assert psfs_dir.exists(), f"psfs dir not found: {psfs_dir}"

        has_gt = lensed_dir.exists()
        if not has_gt:
            logger.warning("No lensed/ directory found — running without ground truth.")

        index = []
        for lensless_path in sorted(lensless_dir.glob("*.png")):
            image_id = lensless_path.stem
            psf_path = psfs_dir / f"{image_id}.npy"

            if not psf_path.exists():
                logger.warning(f"PSF not found for {image_id}, skipping.")
                continue

            lensed_path = lensed_dir / f"{image_id}.png" if has_gt else None

            index.append({
                "path": str(lensless_path),
                "label": 0,
                "lensed_path": str(lensed_path) if lensed_path else None,
                "psf_path": str(psf_path),
                "image_id": image_id,
            })

        logger.info(f"DigiCamDataset ({self.data_dir.name}): {len(index)} samples.")
        return index

    def load_object(self, path: str) -> torch.Tensor:
        return self.to_tensor(Image.open(path).convert("RGB"))

    def _load_gt(self, path: str) -> torch.Tensor:
        return self.to_tensor(Image.open(path).convert("RGB"))

    def _load_psf(self, path: str) -> torch.Tensor:
        return torch.from_numpy(np.load(path).astype(np.float32))

    def __getitem__(self, ind: int) -> dict:
        entry = self._index[ind]

        instance = {
            "lensless": self.load_object(entry["path"]),
            "psf": self._load_psf(entry["psf_path"]),
            "image_id": entry["image_id"],
        }

        if entry["lensed_path"] is not None:
            instance["lensed"] = self._load_gt(entry["lensed_path"])

        return self.preprocess_data(instance)
