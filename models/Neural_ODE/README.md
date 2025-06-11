# Neural ODE and Augmented Neural ODE with CNN Backbone

This directory contains two implementations of models for full-waveform inversion (FWI) using deep learning:

1. **Basic Neural ODE** (in `NeuralODE.ipynb`)
   Uses an encoder-ODE-decoder architecture to map seismic data to a subsurface velocity map. This Jupyter notebook includes both a basic Neural ODE and an Augmented Neural ODE model with training, validation, and visualization code. The results from this notebook have already been generated.

2. **Augmented Neural ODE with CNN Backbones** (in `AugNeuralODECNN.py`)
   A standalone Python script implementing an advanced CNN-augmented Neural ODE model with residual blocks, SE layers, and vectorized ODE integration.

---

## Data Structure

Place the OpenFWI subset in the root directory like this:

```
./
├── FlatVel_A/
│   ├── data/
│   │   └── data1.npy
│   └── model/
│       └── model1.npy
├── CurveFault_A/
│   ├── seis2_1_0.npy
│   └── vel2_1_0.npy
├── FlatFault_A/
│   ├── seis2_1_0.npy
│   └── vel2_1_0.npy
├── Style_A/
│   ├── data/
│   │   └── data1.npy
│   └── model/
│       └── model1.npy
```

---

## Usage

### Neural ODE (Notebook)

1. Open `NeuralODE.ipynb` in Jupyter.
2. Run all cells to:

   * Load and process data
   * Train a basic Neural ODE model
   * Train an augmented version (AugNeuralODE)
   * Visualize predictions
   * Save and test models
3. Results include MAE, RMSE, SSIM, and relative L2 error on the test set.

**Note:** This notebook has already been run and contains results. You can modify it to retrain if desired.

---

### Augmented CNN-based Neural ODE

To train:

```bash
python AugNeuralODECNN.py train
```

To test:

```bash
python AugNeuralODECNN.py test
```

During training:

* Models are saved every 10 epochs as `aug_node_epoch*.pth`
* Validation MAE and denormalized error in m/s are printed

During testing:

* Loads the final model (epoch 150) by default
* Prints MAE on the test split

You may modify the script to change hyperparameters like `BATCH`, `EPOCHS`, learning rate, or model architecture.