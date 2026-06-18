import torch
import numpy as np
import matplotlib.pyplot as plt
from src.datasets.digicam import DigiCamDataset

ds = DigiCamDataset("data/test")
sample = ds[0]

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(sample['lensless'].permute(1, 2, 0))
plt.title("lensless input")
plt.subplot(1, 2, 2)
plt.imshow(sample['lensed'].permute(1, 2, 0))
plt.title("ground truth")
plt.show()
