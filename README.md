# Lensless Computational Imaging

Reconstruction of images captured by a **mask-based lensless camera** (DigiCam). Instead of a lens, the camera uses an
LCD amplitude mask, so the sensor records a blurred, unrecognizable pattern. The original scene is recovered
algorithmically.

This repository implements and compares five reconstruction methods, following:

- Monakhova et al., *Learned reconstructions for practical mask-based lensless imaging* (Le-ADMM).
- Bezzam et al., *Towards Robust and Generalizable Lensless Imaging with Modular Learned Reconstruction* (modular
  pre/post processing).

## Methods

| Method                    | Trainable | Description                                                                               |
|---------------------------|-----------|-------------------------------------------------------------------------------------------|
| **ADMM-100**              | no        | Classical unrolled ADMM, 100 iterations, fixed hyperparameters (μ, τ). TV regularization. |
| **Unrolled ADMM-20**      | yes       | Same algorithm with per-iteration learnable μ, τ (20 iterations).                         |
| **LeADMM-5 (pre + post)** | yes       | DRUNet pre-processor → ADMM-5 → DRUNet post-processor (~8M params).                       |
| **LeADMM-5 (pre only)**   | yes       | DRUNet pre-processor → ADMM-5.                                                            |
| **LeADMM-5 (post only)**  | yes       | ADMM-5 → DRUNet post-processor.                                                           |

The forward model is a shift-invariant convolution with a per-sample PSF, computed efficiently in the Fourier domain.
The PSF is simulated from each raw LCD mask via band-limited angular-spectrum propagation. ADMM operates in a padded
2H×2W space with anisotropic TV and circular finite differences, starting from an all-zeros estimate.

## Installation

```bash
git clone https://github.com/AlexanderPlotnikovv/LenslessComputationalImaging.git
cd LenslessComputationalImaging
pip install -r requirements.txt
```

## Dataset

The project
uses [DigiCam-Mirflickr-MultiMask-10K](https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-MultiMask-10K). Download
and prepare it (downloads images, copies the per-sample masks, and pre-computes PSFs):

```bash
python download_dataset.py --data_dir ./data
```

This produces:

```
data/
  train/   lensless/  lensed/  masks/  psfs/
  test/    lensless/  lensed/  masks/  psfs/
```

## Models

Trained models are hosted on the HuggingFace Hub. Download them with:

```bash
python download_models.py
```

This fetches the four trained models into `models/<model_name>/model_best.pth`.

## Training

Logging is done via Comet ML. Provide your API key when prompted (or set `COMET_API_KEY`).

```bash
# best model: pre + post processors
python3 train.py -cn lensless \
  model=le_admm_pre_post \
  writer.run_name=le_admm_pre_post \
  trainer.device=cuda \
  dataloader.batch_size=16 \
  optimizer.lr=3e-4 \
  loss_function.lpips_weight=0.1 \
  datasets.train.data_dir=./data/train \
  datasets.test.data_dir=./data/test
```

Switch the model by changing `model=` to `le_admm_pre`, `le_admm_post`, or `unrolled_admm20`. ADMM-100 is not trained (
fixed hyperparameters).

## Inference

Reconstruct a dataset and save the predictions. The reconstruction filename matches the input image ID.

```bash
python3 inference.py \
  model=le_admm_pre_post \
  datasets=digicam_eval \
  inferencer.from_pretrained=models/le_admm_pre_post/model_best.pth \
  inferencer.save_path=le_admm_pre_post \
  inferencer.device=cuda \
  ++datasets.test.data_dir=./data/test
```

Reconstructions are written to `data/saved/le_admm_pre_post/test/<ImageID>.png`.

To run on any custom directory (format below), use `datasets=custom_dir` and pass the path:

```bash
python3 inference.py \
  model=le_admm_pre_post \
  datasets=custom_dir \
  +inferencer.data_dir=/path/to/data \
  inferencer.from_pretrained=models/le_admm_pre_post/model_best.pth \
  inferencer.save_path=custom
```

Expected directory format:

```
NameOfTheDirectoryWithData/
  lensless/   ImageID.png ...
  masks/      ImageID.npy ...
  lensed/     ImageID.png ...   # optional ground truth
```

Or you are also can use:

```
./run_inference.sh device /path/to/data
```

## Metrics

Compute PSNR, SSIM, MSE, and LPIPS between reconstructions and ground truth:

```bash
python3 calculate_metrics.py \
  --gt_dir ./data/test/lensed \
  --pred_dir data/saved/le_admm_pre_post/test
```

## Reconstruction speed

Measure per-image reconstruction time for all methods:

```bash
python3 measure_speed.py --data_dir ./data/test
```

## Demo

`demo.ipynb` is a self-contained Colab notebook that clones the repo, downloads the checkpoint, downloads a `.zip`
dataset from a Google Drive URL, runs inference, visualizes samples, and reports metrics. Open it in Colab, set the
dataset URL, and run all cells.

## Project structure

```
src/
  model/        admm.py, drunet.py, le_admm.py
  datasets/     digicam.py, custom_dir_dataset.py
  loss/         combined.py (MSE + LPIPS)
  metrics/      lensless.py (PSNR, SSIM, MSE, LPIPS)
  trainer/      base_trainer.py, trainer.py, inferencer.py
  logger/       cometml.py
  utils/        psf_utils.py (PSF simulation)
  configs/      Hydra configs
train.py
inference.py
calculate_metrics.py
download_dataset.py
download_models.py
demo.ipynb
```