## How to Run PCA Code:  

### Step 0: What Are the Model.py and PCA.py Classes? 

#### PCA.py 
PCA.py contains the PCAStreamlined Class that takes in data loaded in from a .npy file.

Depending on which dataset you are loading (whether it be seismic or velocity) You will use the PCASeismic or PCAVelocity child classes to do so.  

PCASeismic will take a given batch of 5 x 1000 x 70 and PCA it (if told to) when get_batch() is called into a 5000 x n dimension matrix, where n is the number of components to keep as specified in teh constructor. This will then be flattened. 

PCAVelocity will do the same, but instead it will take a 70 x 70 batch size and flatten that instead. 

##### Model.py

Contains the MLP Class and Trainer function. 

### Step 1: 

GetPCAData.py will get the actual flattened PCA data to be fed into the model.  

This will be stored in your local directory in a folder called PCAS into train, test, and validation sets. 

Then GetModel.py will use the flattened PCA data to train the model, which will also be stored in the PCAS folder. 

GetVisData.py will use obtain a new batch for visualizations. 

TLDR: Run GetPCADATA.py-->GetModel.py-->GetVisData.py. 

### Step 2: 

You can then visualize the data by running the code in visualizationpca.py.

