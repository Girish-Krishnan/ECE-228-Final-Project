# PCA-Based Velocity Prediction Pipeline

This directory contains code that applies PCA to seismic and velocity data to compress input dimensions before training a fully connected neural network (MLP) to predict velocity maps.

---

## File Overview

### `PCA.py`

Defines the PCA logic:

* `PCAStreamlined`: Base class that applies PCA to any dataset.
* `PCASeismic`: Child class used to process seismic data of shape `5 x 1000 x 70`. It reshapes and optionally applies PCA, resulting in `5000 x n` features per sample.
* `PCAVelocity`: Child class for velocity data of shape `70 x 70`. This is flattened and optionally PCA-compressed.

### `Model.py`

Defines the model architecture and training:

* `MLP`: A customizable feedforward neural network.
* `trainer`: Function to train and evaluate the MLP model using metrics including MAE, RMSE, SSIM, and relative L2 error.

### `GetPCAData.py`

* Loads raw `.npy` seismic and velocity data from the `FlatVel_A` dataset.
* Applies PCA to both datasets.
* Splits data into train, validation, and test sets.
* Saves them into a directory called `PCAS/` as `.npy` files.

### `GetModel.py`

* Loads the PCA-compressed training data from `PCAS/`.
* Initializes an MLP model.
* Trains it using `trainer()` for 100 epochs.
* Saves the model weights to `PCAS/pca_model.pth`.

### `GetVisData.py`

* Loads a new `.npy` seismic file (`FlatFault_A/seis2_1_0.npy`).
* Applies PCA using `PCASeismic`.
* Saves the compressed batch to `PCAS/VISUALIZATION.npy`.

### `visualizationpca.py`

* Loads `VISUALIZATION.npy` and the trained MLP.
* Runs prediction on the input data.
* Reshapes and visualizes the first predicted velocity map as a `70 x 70` image.

---

## How to Run the Pipeline

### Step 0: Environment Setup

Make sure to install required Python packages:

```bash
pip install numpy torch matplotlib scikit-learn torchmetrics
```

Ensure your working directory has the following structure:

```
./
├── FlatVel_A/
│   ├── data/data1.npy
│   └── model/model1.npy
├── FlatFault_A/
│   └── seis2_1_0.npy
```

### Step 1: Generate PCA Data

Run the following to generate PCA-compressed training data:

```bash
python GetPCAData.py
```

This will create and populate the `PCAS/` directory with:

* `train_X.npy`, `train_Y.npy`
* `val_X.npy`, `val_Y.npy`
* `test_X.npy`, `test_Y.npy`

### Step 2: Train the MLP Model

Run:

```bash
python GetModel.py
```

This will train the model and save it to `PCAS/pca_model.pth`.

### Step 3: Prepare Visualization Input

Run:

```bash
python GetVisData.py
```

This generates `PCAS/VISUALIZATION.npy`, a batch of PCA-compressed seismic data for prediction.

### Step 4: Visualize the Predictions

Finally, run:

```bash
python visualizationpca.py
```

This loads the trained model, makes predictions, and visualizes the first predicted velocity map using `matplotlib`.

---

## Notes

* The PCA components are set to 18 by default. You can adjust this in `GetPCAData.py` and `GetVisData.py`.
* Ensure all `.npy` paths are correct and consistent with your local directory.
* The model assumes that the output velocity shape is square (e.g. `70x70`) to compute SSIM.

---

## TLDR

To reproduce results:

1. Run `GetPCAData.py`
2. Run `GetModel.py`
3. Run `GetVisData.py`
4. Run `visualizationpca.py`

All intermediate outputs and the final model are saved inside the `PCAS/` directory.