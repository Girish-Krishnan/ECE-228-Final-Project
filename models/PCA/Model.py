print("Initiate Loading")
from sklearn.decomposition import PCA 
from sklearn import preprocessing 
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional import mean_absolute_error, mean_squared_error, structural_similarity_index_measure


class MLP(nn.Module):

    def __init__(self,a_input_dim, a_hidden_dims, a_output_dim, activation_func, a_neural_block, a_dropout):
        super(MLP, self).__init__()

        self.my_input_layer=a_neural_block(a_input_dim, a_hidden_dims[0]) 
        self.my_input_norm=nn.LayerNorm(a_input_dim)
        self.my_output_layer=a_neural_block(a_hidden_dims[-1],a_output_dim)
        self.my_layers= nn.ModuleList([a_neural_block(a_hidden_dims[i],a_hidden_dims[i+1]) for i in range(len(a_hidden_dims)-1)]) 
        self.my_layer_norms=nn.ModuleList([nn.LayerNorm(a_hidden_dims[i+1]) for i in range(len(a_hidden_dims)-1)]) 
        self.my_dropout=nn.Dropout(a_dropout)
        self.my_activation=activation_func

    def forward(self, x): 
        x=self.my_dropout(self.my_activation(self.my_input_layer(x)))
        for a_layer, a_norm in zip(self.my_layers,self.my_layer_norms): x=a_norm(self.my_dropout(self.my_activation(a_layer(x)))) 
        x=self.my_output_layer(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

def trainer(model, trainData, valData, testData, learning_rate, step_size, epochs, output_rate):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    loss_fn = torch.nn.MSELoss()

    for ep in range(epochs):
        model.train()
        train_loss = 0.0

        for x_vals, y_vals in trainData:
            x_vals, y_vals = x_vals.to(device), y_vals.to(device)
            optimizer.zero_grad()
            out = model(x_vals)
            loss = loss_fn(out, y_vals)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Evaluation for validation and test sets
        model.eval()

        def eval_metrics(loader):
            eval_loss = 0.0
            eval_mae = 0.0
            eval_rmse = 0.0
            eval_ssim = 0.0
            eval_rel_l2 = 0.0

            with torch.no_grad():
                for x_vals, y_vals in loader:
                    x_vals, y_vals = x_vals.to(device), y_vals.to(device)
                    out = model(x_vals)

                    eval_loss += loss_fn(out, y_vals).item() # mse loss standard, item called to get original python number
                    eval_mae  += mean_absolute_error(out, y_vals).item() # self explanatory
                    eval_rmse += mean_squared_error(out, y_vals, squared=False).item()# self explanatory
                    eval_rel_l2 += (torch.norm(out - y_vals) / torch.norm(y_vals)).item()# self explanatory

                    n, d = out.shape #n = batch size, #d =feature number
                    side = int(d**0.5)
                    if side * side == d:# only works if d is a perfect square, so that ssim can work
                        out_img = out.view(n, 1, side, side) # re view to batch size on perfect square dimensions for ssim
                        y_img = y_vals.view(n, 1, side, side)
                        eval_ssim += structural_similarity_index_measure(out_img, y_img).item()
                    else:
                        eval_ssim += 0.0

            n_batches = len(loader)
            return {
                'loss': eval_loss / n_batches,
                'mae': eval_mae / n_batches,
                'rmse': eval_rmse / n_batches,
                'ssim': eval_ssim / n_batches,
                'rel_l2': eval_rel_l2 / n_batches
            }

        train_loss /= len(trainData)
        
        #eval metrics is a function that takes in a given dataset and calculates the prediction error metrics via forward pass, no gradient descent  
        #that way, we dont need to predict/run trainer twice. 
        #We can instead call it once, now with the val and test sets, and run eval metrics on both instead of one 
        
        val_metrics = eval_metrics(valData) 
        test_metrics = eval_metrics(testData)

        if ep % output_rate == 0:
            print(f"Epoch {ep}: " 
                  f"Val MAE = {val_metrics['mae']:.6f}, "  
                  f"Test MAE = {test_metrics['mae']:.6f}, "
                  f"Test RMSE = {test_metrics['rmse']:.6f}, "
                  f"1-SSIM = {1 - test_metrics['ssim']:.6f}, "
                  f"Rel L2 = {test_metrics['rel_l2']:.6f}")

    return model
