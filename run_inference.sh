DEVICE=${1:-mps}
DATA=${2:-./data/test}

echo "Device: $DEVICE | Data: $DATA"

for model in le_admm_pre_post le_admm_pre le_admm_post unrolled_admm20; do
  echo "=== Inference: $model ==="
  python3 inference.py \
    model=$model \
    datasets=digicam_eval \
    inferencer.from_pretrained=models/$model/model_best.pth \
    inferencer.save_path=$model \
    inferencer.device=$DEVICE \
    dataloader.batch_size=8 \
    dataloader.num_workers=0 \
    ++datasets.test.data_dir=$DATA

  echo "=== Metrics: $model ==="
  python3 calculate_metrics.py \
    --gt_dir $DATA/lensed \
    --pred_dir data/saved/$model/test
done

echo "=== Inference: admm100 ==="
python3 inference.py \
  model=admm100 \
  datasets=digicam_eval \
  inferencer.save_path=admm100 \
  inferencer.device=$DEVICE \
  dataloader.batch_size=8 \
  dataloader.num_workers=0 \
  ++datasets.test.data_dir=$DATA \
  ++inferencer.skip_model_load=true

echo "=== Metrics: admm100 ==="
python3 calculate_metrics.py \
  --gt_dir $DATA/lensed \
  --pred_dir data/saved/admm100/test