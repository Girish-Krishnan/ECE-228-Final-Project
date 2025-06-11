# 2D Fourier Neural Operator

This directory contains code to train a 2D Fourier Neural Operator (FNO) on small subsets of the OpenFWI dataset. The goal is to map seismic recordings to subsurface velocity fields using a spectral convolution-based model.

All the code can be found in `2D_FNO.ipynb`.

---

## Files

* `2D_FNO.ipynb` – Main training, validation, and evaluation script
* `data_loading.py` – External helper script (invoked via `%run`) that sets up dataset paths or preprocessing (user-provided)
* `training_example.py` – Supplementary file for configuration or logging (user-provided)

---

## Dataset Setup

The expected dataset should consist of `.npy` file pairs of the form:

* Seismic input: `(...)/data/dataX.npy` or `(...)/seisX.npy`, shaped `(B, 5, 1000, 70)`
* Velocity target: `(...)/model/modelX.npy` or `(...)/velX.npy`, shaped `(B, 1, 70, 70)` or `(B, 70, 70)`

Example usage in code:

```python
data_pairs = [
    ('FlatVel_A/data/data1.npy', 'FlatVel_A/model/model1.npy'),
    ('CurveFault_A/seis2_1_0.npy', 'CurveFault_A/vel2_1_0.npy'),
    ('FlatFault_A/seis2_1_0.npy', 'FlatFault_A/vel2_1_0.npy'),
    ('Style_A/data/data1.npy', 'Style_A/model/model1.npy')
]
```

---

## Data Pipeline

* `SeismicDataset`: Loads, converts, and batches data for PyTorch
* Data is split 70/20/10 into train, validation, and test sets

```python
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
```

---

## Model Architecture: `FNO`

* Encoder: CNN layers + AdaptiveAvgPool to reduce spatial size to `(70, 70)`
* Spectral Conv Blocks: Series of spectral + pointwise convs, activated by GELU
* Projection: Final conv to regress into a 1-channel velocity map
* Grid Encoding: Positional grid is concatenated to the input features

---

## Training

Set hyperparameters:

```python
learning_rate = 5e-3
epochs = 250
```

Then train the model:

```python
model = FNO(...)
optimizer = torch.optim.Adam(...)
criterion = torch.nn.L1Loss()

for epoch in range(epochs):
    ... # forward, backward, validation
```

Model is saved with:

```python
torch.save(model.state_dict(), "model")
```

---

## Inference & Visualization

Reload the model and run inference:

```python
model = FNO(...)
model.load_state_dict(torch.load("model"))
model.eval()
```

Visualize a few test predictions:

```python
plt.imshow(targets[i][0].cpu(), cmap='jet')     # Ground truth
plt.imshow(predictions[i][0].cpu(), cmap='jet')  # Prediction
```

---

## Evaluation Metrics

After training, evaluate on the test set:

* **MAE** (Mean Absolute Error)
* **RMSE** (Root Mean Squared Error)
* **SSIM** (Structural Similarity Index)
* **Relative L2 Error**

All metrics are averaged over the test set and printed to console.

---

## Notes

* This implementation is designed for small subsets of OpenFWI and may not generalize well unless trained on larger datasets
* The model uses adaptive pooling and bilinear upsampling to ensure fixed output size `(70, 70)`
* SSIM is computed using `torchmetrics.functional.structural_similarity_index_measure`

---

## Output

Trained model: `model` (PyTorch state dict)

Visuals and logs: Printed to console; you can redirect or log them using standard Python logging tools if needed