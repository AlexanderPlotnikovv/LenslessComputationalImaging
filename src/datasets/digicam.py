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
        lensed_dir   = self.data_dir / "lensed"
        masks_dir    = self.data_dir / "masks"

        assert lensless_dir.exists(), f"lensless dir not found: {lensless_dir}"
        assert masks_dir.exists(),    f"masks dir not found: {masks_dir}"

        has_gt = lensed_dir.exists()
        if not has_gt:
            logger.warning("No lensed/ directory found — running without ground truth.")

        index = []
        for lensless_path in sorted(lensless_dir.glob("*.png")):
            image_id = lensless_path.stem
            mask_path = masks_dir / f"{image_id}.npy"

            if not mask_path.exists():
                logger.warning(f"Mask not found for {image_id}, skipping.")
                continue

            lensed_path = lensed_dir / f"{image_id}.png" if has_gt else None
            index.append({
                "path":        str(lensless_path),
                "label":       0,
                "lensed_path": str(lensed_path) if lensed_path else None,
                "mask_path":   str(mask_path),
                "image_id":    image_id,
            })

        logger.info(f"DigiCamDataset ({self.data_dir.name}): {len(index)} samples.")
        return index

    def load_object(self, path: str) -> torch.Tensor:
        return self.to_tensor(Image.open(path).convert("RGB"))

    def _load_gt(self, path: str) -> torch.Tensor:
        return self.to_tensor(Image.open(path).convert("RGB"))

    def _load_mask(self, path: str) -> torch.Tensor:
        mask = np.load(path).astype(np.float32)
        t = torch.from_numpy(mask)
        if t.dim() == 2:
            t = t.unsqueeze(0).repeat(3, 1, 1)
        elif t.dim() == 3 and t.shape[0] not in (1, 3):
            t = t.permute(2, 0, 1)
        return t

    def __getitem__(self, ind: int) -> dict:
        entry = self._index[ind]

        lensless = self.load_object(entry["path"])
        mask     = self._load_mask(entry["mask_path"])

        instance = {
            "lensless": lensless,
            "mask":     mask,
            "image_id": entry["image_id"],
        }

        if entry["lensed_path"] is not None:
            instance["lensed"] = self._load_gt(entry["lensed_path"])

        instance = self.preprocess_data(instance)
        return instance