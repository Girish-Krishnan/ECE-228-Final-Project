# VelocityGAN

This directory provides PyTorch implementations for training and evaluating a GAN-based neural network (VelocityGAN) that maps 2D seismic data to velocity fields, using subsets of the OpenFWI dataset.

---

## Overview

VelocityGAN consists of two main scripts:

* `velocity_gan_train.py`: Trains the generator and discriminator networks using adversarial and L1 losses.
* `velocity_gan_eval.py`: Loads a trained generator and evaluates it on the test split, computing MAE, RMSE, SSIM, and relative L2 error.

---

## Dataset Structure

The dataset should follow this structure:

```
./data/
  FlatVel_A/
    data/dataX.npy
    model/modelX.npy
  CurveFault_A/
    seisX.npy
    velX.npy
  FlatFault_A/
    seisX.npy
    velX.npy
  Style_A/
    data/dataX.npy
    model/modelX.npy
```

* Seismic input: Shape `(B, 5, 1000, 70)`
* Velocity target: Shape `(B, 1, 70, 70)` or `(B, 70, 70)`

---

## Training

Run the training script:

```bash
python velocity_gan_train.py --data_root ./data --epochs 50 --batch 8
```

* Model checkpoints will be saved every 5 epochs as `velocitygan_epochX.pt`
* Losses (L1 and adversarial) are printed per iteration

---

## Evaluation

Evaluate a trained model checkpoint with:

```bash
python velocity_gan_eval.py --ckpt velocitygan_epoch50.pt --data_root ./data --batch 8
```

Outputs:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Structural Similarity Index (SSIM)
* Relative L2 error

---

## Notes

* Velocity fields are normalized between \[-1, 1] before training
* Evaluation denormalizes to physical units (m/s)
* Checkpoints contain both Generator (`G`) and Discriminator (`D`) state dicts
* Only `G` is used during evaluation
