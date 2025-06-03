from PCA import PCASeismic
import numpy as np

the_folder='PCAS'
the_seis=PCASeismic(np.load('FlatFault_A/seis2_1_0.npy'),18) 
VISUALIZATION_DATA=the_seis.get_batch(a_pca=True) 
np.save(f'{the_folder}/VISUALIZATION.npy',VISUALIZATION_DATA)