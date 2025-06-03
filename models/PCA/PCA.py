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

class PCAStreamLined:
    def __init__(self, a_data, a_components=0.95): 
        self.my_data=a_data  
        self.my_pca=None 
        self.my_components=a_components 
        self.my_scaler=None 
    
    def show_scree_plot(self):  
        plt.figure()
        the_var_per_pc=np.round(self.my_pca.explained_variance_ratio_*100, decimals =1)
        the_var_per_labels=['PC' + str(a_PC) for a_PC in range(1, len(the_var_per_pc)+1)]
        plt.bar(x=range(1,len(the_var_per_pc)+1), height=the_var_per_pc, tick_label=the_var_per_labels) 
        plt.ylabel('Percentage Of Explained Variance') 
        plt.xlabel('Principal Component') 
        plt.title('Scree Plot') 
        plt.show()   

    def show_data(self, a_data, a_color):  
        plt.figure()
        plt.imshow(
            a_data,
            aspect='auto',
            cmap=a_color,
        ) 

    def perform_pca(self, a_data):
        self.my_scaler=preprocessing.StandardScaler().fit(a_data)  
        the_scaled_data=self.my_scaler.transform(a_data)
        self.my_pca=PCA(n_components=self.my_components) 
        self.my_pca.fit(the_scaled_data) 
        return self.my_pca.transform(the_scaled_data)  

    def reverse_pca(self, a_data): 
        return self.my_scaler.inverse_transform(self.my_pca.inverse_transform(a_data) ) 

    def sanity_check(self, a_batch, a_color):
        the_sanity=self.reverse_pca(self.perform_pca(a_batch))
        self.show_scree_plot()
        self.show_data(the_sanity, a_color)  

    def data_point_at(self, a_sample): pass

        
    def get_batch(self, a_pca=False):  
        return np.array([self.perform_pca(self.data_point_at(i)).flatten() if a_pca else self.data_point_at(i).flatten() for i in range(self.my_data.shape[0]) ])

class PCASeismic(PCAStreamLined): 
    def __init__(self, a_data, a_components): super().__init__(a_data, a_components)
    def data_point_at(self, a_sample): return np.vstack([self.my_data[a_sample, i , :, :].reshape(self.my_data.shape[2],self.my_data.shape[3]) for i in range(self.my_data.shape[1])])
    def sanity_check(self, a_sample):super().sanity_check(self.data_point_at(a_sample), 'gray') 
      
        
class PCAVelocity(PCAStreamLined): 
    def __init__(self, a_data, a_components): super().__init__(a_data, a_components)
    def data_point_at(self, a_sample): return self.my_data[a_sample, :, :].reshape(self.my_data.shape[2],self.my_data.shape[3])   
    def sanity_check(self, a_sample):super().sanity_check(self.data_point_at(a_sample), 'jet') 
