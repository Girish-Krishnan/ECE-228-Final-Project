## How to Run PCA Code

### Step 0: What Are the `Model.py` and `PCA.py` Classes?

#### `PCA.py`

`PCA.py` contains the `PCAStreamlined` class that takes in data loaded from a `.npy` file.

Depending on which dataset you are loading (whether it be seismic or velocity), you will use the `PCASeismic` or `PCAVelocity` child classes to do so.

`PCASeismic` will take a given batch of size `5 x 1000 x 70` and PCA it (if told to) when `get_batch()` is called. It reshapes the data into a `5000 x n` dimension matrix, where `n` is the number of components to keep as specified in the constructor. This result is then flattened.

`PCAVelocity` will do the same, but instead it takes a `70 x 70` batch size and flattens that instead.

#### `Model.py`

Contains the `MLP` class and `Trainer` function.

---

### Step 1: Running the Pipeline

`GetPCAData.py` will get the actual flattened PCA data to be fed into the model.

This data will be stored in your local directory inside a folder called `PCAS`, split into train, test, and validation sets.

Then `GetModel.py` will use the flattened PCA data to train the model, which will also be stored in the `PCAS` folder.

`GetVisData.py` will obtain a new batch for visualizations.

**TLDR:** Run the following in order:

* `GetPCAData.py`
* `GetModel.py`
* `GetVisData.py`

---

### Step 2: Visualize

You can then visualize the data by running the code in `visualizationpca.py`.
