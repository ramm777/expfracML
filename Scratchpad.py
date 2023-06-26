# Some random scripts

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg') # For the saving purposes 'Agg'. To plot online 'TkAgg'
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split

import keras
import keras.models as km
from keras.preprocessing.image import ImageDataGenerator

import pandas as pd
from pathlib import Path

import time
import utils as ff
import cnn_architectures as ff1
import plot_functions as vis

#-----------------------------------------------------------------------------------------------------------------------
# Scripts to calculate computation time of CNN vs Numerical Simulaiton. The numerical part is called time_comp.m
#-----------------------------------------------------------------------------------------------------------------------

# Select manually:
path_test = Path("data/Carmel_nodepth_test2000/Augmented_centered/Bald") # Test X and y data, original
path_train = Path("data/Train/Augmented_centered/Bald")
modelname = "model_cnn6_19.h5py"
modelpath = Path("selected_models/Bald/")
imsize_x = 128
imsize_y = 128

path_traintest = path_train
datapath = path_test

test_X = np.load(datapath / "test_X.npy" )
test_X = test_X.astype('float32')
test_X = test_X.reshape(-1, imsize_x, imsize_y, 1)
test_X = test_X / 255.

#test_Y = np.loadtxt(datapath / "permf.csv")
#test_Y = test_Y / 1e4  # convert to 'mD/1e4'
#test_Y = test_Y.reshape(-1, 1)
#test_Y_scaled = scaler.transform(test_Y)

# Select only one fracture
test_X = test_X[4, :, :, :].copy()
test_X = test_X.reshape(-1, imsize_x, imsize_y, 1)

if False:
    fig, (ax0, ax1) = plt.subplots(1,2)
    ax0.imshow(test_X[4, :, :, :])
    ax1.imshow(test_X[-1, :, :, :])

model = km.load_model((modelpath / modelname), custom_objects=None, compile=True)

#-----------------------------------------------------------------------------------------------------------------------
# Measure time from here

n_runs = 40

start = time.time()
for j in range(n_runs):
    scaler = ff.getScaler(path_traintest)  # Scale from 0 to 1
    predicted = model.predict(test_X)
    predicted_unscaled = scaler.inverse_transform(predicted)  # to unscale the data back
    predicted_unscaled = predicted_unscaled * 10
print('Time, min: ',  (time.time() - start)/60 )

#-----------------------------------------------------------------------------------------------------------------------
# Plot

# !pip install openpyxl
time = pd.read_excel('comp_time.xlsx')
fig, ax = plt.subplots()
ax.plot(time['n_runs'], time['time_numerical_min'], '-o')
ax.plot(time['n_runs'], time['time_cnn_min'], '-o')
plt.xlabel('Number of iterations', fontsize=16)
plt.ylabel('Computation time, min', fontsize=16)
plt.legend(['Numerical Simulation', 'Convolutional Neural Network'], frameon=False)




# Maunally divide test / valid as 222*30 and 25*30
train_S1 = train_S[0 : 247*30].copy()
train_S2 = train_S[247*30 : 247*30*2].copy()
train_S3 = train_S[247*30*2 : 247*30*3].copy()
train_S4 = train_S[247*30*3 : 247*30*4].copy()

Train_s1 = train_S1[0:6660].copy()
Train_s2 = train_S2[0:6660].copy()
Train_s3 = train_S3[0:6660].copy()
Train_s4 = train_S4[0:6660].copy()

Valid_s1 = train_S1[6660:7410].copy()
Valid_s2 = train_S2[6660:7410].copy()
Valid_s3 = train_S3[6660:7410].copy()
Valid_s4 = train_S4[6660:7410].copy()

Train_s = np.concatenate((Train_s1, Train_s2, Train_s3, Train_s4))
Valid_s = np.concatenate((Valid_s1, Valid_s2, Valid_s3, Valid_s4))
del train_S1, train_S2, train_S3, train_S4, Train_s1, Train_s2, Train_s3, Train_s4, Valid_s1, Valid_s2, Valid_s3, Valid_s4

train_Y1 = train_Y[0 : 247*30].copy()
train_Y2 = train_Y[247*30 : 247*30*2].copy()
train_Y3 = train_Y[247*30*2 : 247*30*3].copy()
train_Y4 = train_Y[247*30*3 : 247*30*4].copy()

Train_y1 = train_Y1[0:6660].copy()
Train_y2 = train_Y2[0:6660].copy()
Train_y3 = train_Y3[0:6660].copy()
Train_y4 = train_Y4[0:6660].copy()

Valid_y1 = train_Y1[6660:7410].copy()
Valid_y2 = train_Y2[6660:7410].copy()
Valid_y3 = train_Y3[6660:7410].copy()
Valid_y4 = train_Y4[6660:7410].copy()

Train_y = np.concatenate((Train_y1, Train_y2, Train_y3, Train_y4))
Valid_y = np.concatenate((Valid_y1, Valid_y2, Valid_y3, Valid_y4))
del train_Y1, train_Y2, train_Y3, train_Y4, Train_y1, Train_y2, Train_y3, Train_y4, Valid_y1, Valid_y2, Valid_y3, Valid_y4

train_X1 = train_X[0: 247 * 30].copy()
train_X2 = train_X[247 * 30: 247 * 30 * 2].copy()
train_X3 = train_X[247 * 30 * 2: 247 * 30 * 3].copy()
train_X4 = train_X[247 * 30 * 3: 247 * 30 * 4].copy()

Train_x1 = train_X1[0:6660].copy()
Train_x2 = train_X2[0:6660].copy()
Train_x3 = train_X3[0:6660].copy()
Train_x4 = train_X4[0:6660].copy()

Valid_x1 = train_X1[6660:7410].copy()
Valid_x2 = train_X2[6660:7410].copy()
Valid_x3 = train_X3[6660:7410].copy()
Valid_x4 = train_X4[6660:7410].copy()

Train_x = np.concatenate((Train_x1, Train_x2, Train_x3, Train_x4))
Valid_x = np.concatenate((Valid_x1, Valid_x2, Valid_x3, Valid_x4))
del train_X1, train_X2, train_X3, train_X4, Train_x1, Train_x2, Train_x3, Train_x4, Valid_x1, Valid_x2, Valid_x3, Valid_x4

