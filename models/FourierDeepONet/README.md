# Fourier DeepONet

This directory has code that trains a Fourier DeepONet model on OpenFWI-style seismic datasets and provides scripts for evaluation and visualization of prediction performance.

---

## Overview

### Scripts

* `fourier_deeponet.py`: Trains the Fourier DeepONet model on seismic-to-velocity data.
* `fourier_eval.py`: Loads a trained model checkpoint and generates a set of evaluation figures, including prediction error maps and spectral analysis.

### Data Format

The data should follow this directory structure:

```
data/
 ├── FlatVel_A/
 │   ├── data/data1.npy         # shape (N, 5, 1000, 70)
 │   └── model/model1.npy       # shape (N, 1, 70, 70) or (N, 70, 70)
 ├── CurveFault_A/
 │   ├── seis2_1_0.npy
 │   └── vel2_1_0.npy
```

Each `.npy` file pair corresponds to a seismic input and its ground truth velocity map.

---

## Training the Model

Run the following to train the Fourier DeepONet model:

```bash
python fourier_deeponet.py
```

### Training Details

* Splits data into 70% train, 20% validation, 10% test.
* Trains for 100 epochs with L1 loss.
* Every 5 epochs, a checkpoint is saved as `fourier_deeponet_epochXX.pt`.
* Velocity maps are normalized to \[-1, 1] using the physical range \[1500, 4500] m/s.

---

## Evaluating a Trained Model

After training, evaluate and visualize the model:

```bash
python fourier_eval.py \
    --ckpt path/to/fourier_deeponet_epochXX.pt \
    --data_dir ./data \
    --num 5 \
    --outdir ./viz_out
```

### Outputs

* **sampleXX\_triptych.png**: Input gather, predicted velocity, ground truth, and error.
* **scatter\_err\_vs\_vel.png**: Error vs. true velocity density plot.
* **depth\_profile.png**: Mean absolute error as a function of depth.
* **residual\_spectrum.png**: 2D Fourier amplitude spectrum of residuals.

These files are saved in the specified `--outdir` folder.

---

## Model Architecture

The Fourier DeepONet model consists of:

* **Branch Net**: Processes the seismic input using padding and an MLP.
* **Trunk Net**: Encodes dummy scalar inputs.
* **Decoder**: Applies spectral convolutions, U-Net-style residual blocks, and projection layers to predict the velocity map.

---

## Notes

* Both training and evaluation scripts expect `.npy` files laid out in OpenFWI-style directories.
* Trained models are checkpointed in PyTorch format (`.pt`).
* All visualization figures are generated using Matplotlib and saved as `.png`.