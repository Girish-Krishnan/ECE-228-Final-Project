# Residual UNet

This directory contains training and evaluation code for a Residual UNet model applied to seismic-to-velocity inversion using the OpenFWI dataset. The model takes 2D seismic recordings as input and predicts the corresponding 2D subsurface velocity field. It uses a UNet-style encoder-decoder with residual blocks and downsampling via pooling and upsampling via bilinear interpolation.

---

## Files

* `residual_unet_train.py` – Script for training the Residual UNet
* `residual_unet_eval.py` – Script for evaluating the model and visualizing predictions

---

## Dataset Format

The code assumes the dataset is organized with pairs of `.npy` files:

* **Seismic inputs**: shape `(B, 5, 1000, 70)`
* **Velocity maps**: shape `(B, 1, 70, 70)` or `(B, 70, 70)`

Folder structure:

```
./data/
  FlatVel_A/data/*.npy     and FlatVel_A/model/*.npy
  Style_A/data/*.npy       and Style_A/model/*.npy
  CurveFault_A/seis*.npy   and CurveFault_A/vel*.npy
  FlatFault_A/seis*.npy    and FlatFault_A/vel*.npy
```

---

## Training

Run the training script with:

```bash
python residual_unet_train.py
```

The training process includes:

* 70/20/10 train/val/test split
* Training for 100 epochs
* Logging train/val MAE each epoch
* Saving a checkpoint every 5 epochs as `resunet_epoch_XX.pth`

---

## Evaluation

To evaluate a trained model (e.g., epoch 100):

```bash
python residual_unet_eval.py
```

The evaluation script:

* Loads `resunet_epoch_100.pth`
* Computes metrics on the 10% test split:

  * MAE (Mean Absolute Error)
  * RMSE (Root Mean Squared Error)
  * SSIM (Structural Similarity)
  * Relative L2 error
* Visualizes 5 random test predictions alongside ground truth

---

## Notes

* Input and output velocities are normalized during training
* SSIM is computed using `pytorch-msssim`
* The model uses average pooling to reduce the initial vertical dimension