import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import utils as ff


#-----------------------------------------------------------------------------------------------------------------------
# Modify existing images to remove velocity field, substituting it with a single non zero value


datapath_x = Path("data/Train/Augmented_centered/")     # Augmented, centered
dataname = "train_X.npy"

data = np.load(datapath_x / dataname)

new_data = data.copy()
new_data[new_data > 0] = 255

np.save('train_X_bold.npy', new_data)


#-----------------------------------------------------------------------------------------------------------------------
# Data engineering


# Run data augmentation
#datapath_y = "D:\\expfracPython\\data_recognizeStokes\\Train\\Augmented\\permf.csv"     # Augmented
#datapath_x = "D:\\expfracPython\\data_recognizeStokes\\Train\\Augmented\\train_X.npy"   # Augmented
#train_Y = np.loadtxt(datapath_y)
#train_X = np.load(datapath_x)
#train_X, train_Y = ff.dataAugmentation(train_X, train_Y)
#np.save('train_X.npy', train_X)
#np.save('train_Y.npy', train_Y)


# Run centering scripts
#train_X = np.load("D:\\expfracPython\\data_recognizeStokes\\Augmented\\train_X.npy")  # Augmented
#new_images = ff.runCenteringAndPlot(train_X, plot=False)
#np.save('new_images.npy', new_images)


# Run data processing, augmentation and centering
#imnum2 = 2000
#datapath_y = "D:\\expfracPython\\data_recognizeStokes\\Test2000\\permf.csv"     # Augmented
#datapath_x = "D:\\expfracPython\\data_recognizeStokes\\Test2000\\"
#train_Y = np.loadtxt(datapath_y)

#train_X = ff.loadPreprocessImages(datapath_x, imnum2, 128, 128)
#train_X, train_Y = ff.dataAugmentation(train_X, train_Y)
#new_images = ff.runCenteringAndPlot(train_X, plot=False)
#np.save('train_X.npy', new_images)