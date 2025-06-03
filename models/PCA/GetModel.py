import numpy as np 
from Model import MLP, trainer 
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)
print('getting datasets')

the_folder="PCAS"
batch_size=16

X_train = np.load(f'{the_folder}/train_X.npy') 
X_test = np.load(f'{the_folder}/test_X.npy') 
Y_train = np.load(f'{the_folder}/train_Y.npy') 
Y_test = np.load(f'{the_folder}/test_Y.npy')
X_val = np.load(f'{the_folder}/val_X.npy') 
Y_val = np.load(f'{the_folder}/val_Y.npy') 


X_train_tensor = torch.from_numpy(X_train).float()
Y_train_tensor = torch.from_numpy(Y_train).float()
X_test_tensor  = torch.from_numpy(X_test).float()
Y_test_tensor  = torch.from_numpy(Y_test).float()
X_val_tensor  = torch.from_numpy(X_val).float()
Y_val_tensor  = torch.from_numpy(Y_val).float()


train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset  = TensorDataset(X_test_tensor, Y_test_tensor)
test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

val_dataset  = TensorDataset(X_val_tensor, Y_val_tensor)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


print("Training Started")
epochs = 100
learning_rate = 0.15
step_size = 15
output_rate = 1 
activation_func=nn.LeakyReLU()
model=MLP(X_train.shape[1],[2**11, 2**9, 2**7], Y_train.shape[1], activation_func, nn.Linear,0.05).to(device) 
model = trainer(model, train_loader, val_loader, test_loader, learning_rate, step_size, epochs, output_rate)
torch.save(model.state_dict(), f'{the_folder}/pca_model.pth')
