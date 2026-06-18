from src.datasets.digicam import DigiCamDataset
ds = DigiCamDataset("data/train")
print(ds[0].keys())  # должно быть lensless, lensed, mask

sample = ds[0]
print(sample['lensless'].shape)  # должно быть (3, H, W)
print(sample['lensed'].shape)    # должно быть (3, H, W)
print(sample['psf'].shape)     # должно быть (3, H, W) или (1, H, W)
print(sample['image_id'])        # строка типа '00000'