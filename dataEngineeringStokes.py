#-----------------------------------------------------------------------------------------------------------------------
# Data engineering for the recognize Stokes

import numpy as np
from pathlib import Path
import scipy.io as sio
import matplotlib.pyplot as plt
import os

import utils as ff


# Run data processing, augmentation and centering
imnum = 2000
datapath = Path("D:/expfracML/data/Carmel_nodepth_test2000/")
filename_x = "train_X.npy"  # to be created based on the images
filename_y = 'train_Y.npy'
filename_y1 = "permf.mat"   # permeability matlab file
filename_y2 = "permf.csv"   # converted from matlab to csv

train_X = ff.loadPreprocessImages(datapath, imnum, 128, 128)
np.save(filename_x, train_X)
print('Finished loading and processing data')
input("Press Space to continue...")


# Check your data manually
train_X = np.load(filename_x)
plt.imshow(train_X[0, :, :])
plt.colorbar()
input("Press Space to continue...")
plt.close()
del train_X

# Move created file to the datapath_x
Path(filename_x).rename(datapath / filename_x)


# Load permf data
permf = sio.loadmat(datapath / filename_y1)['permf']
np.savetxt(datapath / filename_y2, permf, delimiter=',')


# Run data augmentation
train_X = np.load(datapath / filename_x)
train_Y = np.loadtxt(datapath / filename_y2, delimiter=',')
assert len(train_Y) == 2000 and len(train_X)

train_X, train_Y = ff.dataAugmentation(train_X, train_Y)

os.mkdir(datapath / "Augmented")
print("Directory (folder) created: %s" % "Augmented")
np.save(datapath / "Augmented" / filename_x, train_X)
np.save(datapath / "Augmented" / filename_y, train_Y)
np.savetxt(datapath / "Augmented" / filename_y2, train_Y, delimiter=',')
del train_X, train_Y


# Run image centering on augmented image data
train_X = np.load(datapath / "Augmented" / filename_x)
train_Y = np.load(datapath / "Augmented" / filename_y)
new_images = ff.runCenteringAndPlot(train_X, plot=False)

os.mkdir(datapath / "Augmented_centered")
np.save(datapath / "Augmented_centered" / filename_x, new_images)
np.save(datapath / "Augmented_centered" / filename_y, train_Y)



#-----------------------------------------------------------------------------------------------------------------------
# Load and concatinate permf arrays of 70000 images


#permf1 = sio.loadmat("D:\\expfracML\\data\\TrainTest75000\\permf\\permf1.mat")['permf']
#permf2 = sio.loadmat("D:\\expfracML\\data\\TrainTest75000\\permf\\permf2.mat")['permf']
#permf3 = sio.loadmat("D:\\expfracML\\data\\TrainTest75000\\permf\\permf3.mat")['permf']
#permf4 = sio.loadmat("D:\\expfracML\\data\\TrainTest75000\\permf\\permf4.mat")['permf']
#permf5 = sio.loadmat("D:\\expfracML\\data\\TrainTest75000\\permf\\permf5.mat")['permf']
#permf6 = sio.loadmat("D:\\expfracML\\data\\TrainTest75000\\permf\\permf6.mat")['permf']
#permf7 = sio.loadmat("D:\\expfracML\\data\\TrainTest75000\\permf\\permf7.mat")['permf']
#permf8 = sio.loadmat("D:\\expfracML\\data\\TrainTest75000\\permf\\permf8.mat")['permf']
#permf9 = sio.loadmat("D:\\expfracML\\data\\TrainTest75000\\permf\\permf9.mat")['permf']
#permf10 = sio.loadmat("D:\\expfracML\\data\\TrainTest75000\\permf\\permf10.mat")['permf']
#permf11 = sio.loadmat("D:\\expfracML\\data\\TrainTest75000\\permf\\permf11.mat")['permf']
#permf12 = sio.loadmat("D:\\expfracML\\data\\TrainTest75000\\permf\\permf12.mat")['permf']
#permf13 = sio.loadmat("D:\\expfracML\\data\\TrainTest75000\\permf\\permf13.mat")['permf']
#permf14 = sio.loadmat("D:\\expfracML\\data\\TrainTest75000\\permf\\permf14.mat")['permf']
#permf15 = sio.loadmat("D:\\expfracML\\data\\TrainTest75000\\permf\\permf15.mat")['permf']


#permf1_1 = permf1[0, :]
#permf2_1 = permf2[0, 5000:]
#permf3_1 = permf3[0, 10000:]
#permf4_1 = permf4[0, 15000:]
#permf5_1 = permf5[0, 20000:]
#permf6_1 = permf6[0, 25000:]
#permf7_1 = permf7[0, 30000:]
#permf8_1 = permf8[0, 35000:]
#permf9_1 = permf9[0, 40000:]
#permf10_1 = permf10[0, 45000:]
#permf11_1 = permf11[0, 50000:]
#permf12_1 = permf12[0, 55000:]
#permf13_1 = permf13[0, 60000:]
#permf14_1 = permf14[0, 65000:]
#permf15_1 = permf15[0, 70000:]

#del permf1, permf2, permf3, permf4, permf5, permf6, permf7, permf8, permf9, permf10, permf11, permf12, permf13, permf14, permf15
#permf = np.hstack((permf1_1, permf2_1, permf3_1, permf4_1, permf5_1, permf6_1, permf7_1, permf8_1, permf9_1, permf10_1, permf11_1, permf12_1, permf13_1, permf14_1, permf15_1))


