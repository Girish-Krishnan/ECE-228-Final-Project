from PCA import PCASeismic, PCAVelocity
import numpy as np
from sklearn.model_selection import train_test_split


batch_size = 64

seismic_data = np.load('FlatVel_A/data/data1.npy', allow_pickle=True) 
velocity = np.load('FlatVel_A/model/model1.npy', allow_pickle=True)


the_seis=PCASeismic(seismic_data,a_components=18) 
the_vel=PCAVelocity(velocity,a_components=18) 

print("run sanity check")
#the_seis.sanity_check(10)
#the_vel.sanity_check(10) 

print("Performing PCA")
X=the_seis.get_batch(a_pca=True) 
Y=the_vel.get_batch()
print('done pca')

print('splitting')
X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Now split train+val into train and val
X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size=0.2, random_state=42)

the_folder="PCAS"

np.save(f'{the_folder}/train_X.npy', X_train)
np.save(f'{the_folder}/val_X.npy', X_val)
np.save(f'{the_folder}/test_X.npy', X_test)
np.save(f'{the_folder}/train_Y.npy', Y_train)
np.save(f'{the_folder}/val_Y.npy', Y_val)
np.save(f'{the_folder}/test_Y.npy', Y_test)



    