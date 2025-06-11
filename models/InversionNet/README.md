# InversionNet

This directory contains code to train and evaluate **InversionNet**, a deep convolutional network for seismic-to-velocity inversion on small subsets of the OpenFWI dataset.

---

## Files Overview

* `inversion_net.py`: Main training script for InversionNet
* `inversion_eval.py`: Inference and evaluation script computing MAE, RMSE, SSIM, and Relative L2 error

---

## Dataset Format

Each sample consists of:

* Seismic Input: shape `(B, 5, 1000, 70)`
* Velocity Ground Truth: shape `(B, 1, 70, 70)` or `(B, 70, 70)`

Expected folder layout:

```
./data/
  FlatVel_A/data/*.npy     and FlatVel_A/model/*.npy
  Style_A/data/*.npy       and Style_A/model/*.npy
  CurveFault_A/seis*.npy   and CurveFault_A/vel*.npy
  FlatFault_A/seis*.npy    and FlatFault_A/vel*.npy
```

---

## Training

To train the InversionNet model:

```bash
python inversion_net.py \
    --root ./data \
    --epochs 100 \
    --bs 8 \
    --ckpt inversionnet_epoch
```

This will train for 100 epochs and save model checkpoints every 5 epochs to `inversionnet_epoch_XX.pt`.

---

## Evaluation

To evaluate a saved model checkpoint and visualize predictions:

```bash
python inversion_eval.py \
    --ckpt inversionnet_epoch_100.pt \
    --data_dir ./data \
    --n 5
```

This script will:

* Compute evaluation metrics: MAE, RMSE, SSIM, Relative L2 error
* Plot predictions for `n` test samples alongside ground truth

---

## Notes

* Velocity values are normalized to `[-1, 1]` for training, and de-normalized for evaluation.
* The model uses a U-Net-like encoder-decoder architecture with skip connections removed.
* Evaluation is done on a 10% test split from the dataset.
