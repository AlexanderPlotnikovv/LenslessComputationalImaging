import torch


def collate_fn(dataset_items: list[dict]):
    result_batch = {
        "lensless": torch.stack([item["lensless"] for item in dataset_items]),
        "psf": torch.stack([item["psf"] for item in dataset_items]),
        "image_id": [item["image_id"] for item in dataset_items],
    }

    if "lensed" in dataset_items[0]:
        result_batch["lensed"] = torch.stack([item["lensed"] for item in dataset_items])
    return result_batch
