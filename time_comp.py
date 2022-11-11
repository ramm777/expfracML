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
import CNNarchitectures as ff1
import plotFunctions as vis

import recognizeStokes as rs


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
ax.plot(time['n_runs'], time['time_numerical_min'], '-x')
ax.plot(time['n_runs'], time['time_cnn_min'], '-x')
plt.xlabel('Number of iterations')
plt.ylabel('Computation time, min')

