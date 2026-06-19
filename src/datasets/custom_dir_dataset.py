import logging
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from src.datasets.base_dataset import BaseDataset
from src.utils.psf_utils import simulate_psf

logger = logging.getLogger(__name__)


class CustomDirDataset(BaseDataset):
    def __init__(self, data_dir: str, image_size: tuple = (270, 360), *args, **kwargs):
        self.data_dir = Path(data_dir)
        self.image_size = tuple(image_size)
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])
        index = self._create_index()
        super().__init__(index, *args, **kwargs)

    def _create_index(self):
        lensless_dir = self.data_dir / "lensless"
        masks_dir = self.data_dir / "masks"
        lensed_dir = self.data_dir / "lensed"

        assert lensless_dir.exists(), f"lensless/ not found in {self.data_dir}"
        assert masks_dir.exists(), f"masks/ not found in {self.data_dir}"

        has_gt = lensed_dir.exists()
        index = []
        for lensless_path in sorted(lensless_dir.iterdir()):
            if lensless_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            image_id = lensless_path.stem
            mask_path = masks_dir / f"{image_id}.npy"
            if not mask_path.exists():
                logger.warning(f"No mask for {image_id}, skipping.")
                continue
            lensed_path = lensed_dir / f"{image_id}.png" if has_gt else None
            index.append({
                "path": str(lensless_path),
                "label": 0,
                "lensed_path": str(lensed_path) if lensed_path else None,
                "mask_path": str(mask_path),
                "image_id": image_id,
            })
        logger.info(f"CustomDirDataset: {len(index)} samples from {self.data_dir}.")
        return index

    def load_object(self, path: str) -> torch.Tensor:
        return self.to_tensor(Image.open(path).convert("RGB"))

    def _load_gt(self, path: str) -> torch.Tensor:
        return self.to_tensor(Image.open(path).convert("RGB"))

    def _load_psf(self, path: str) -> torch.Tensor:
        mask = np.load(path)
        return simulate_psf(mask, self.image_size)

    def __getitem__(self, ind: int) -> dict:
        entry = self._index[ind]
        instance = {
            "lensless": self.load_object(entry["path"]),
            "psf": self._load_psf(entry["mask_path"]),
            "image_id": entry["image_id"],
        }
        if entry["lensed_path"] is not None:
            instance["lensed"] = self._load_gt(entry["lensed_path"])
        return self.preprocess_data(instance)
