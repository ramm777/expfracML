import numpy as np
from pathlib import Path
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import utils as ff


# Select manuaally:
imnum = 2000
datapath = Path("D:/expfracML/data/Carmel_nodepth_test2000/")
filename_x = "train_X.npy"  # to be created based on the images
filename_y = 'train_Y.npy'  # to be created permeability
filename_y1 = "permf.mat"   # permeability matlab file
filename_y2 = "permf.csv"   # converted from matlab to csv

# Run data processing, augmentation and centering
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

permf

#-----------------------------------------------------------------------------------------------------------------------
# Load and concatinate permf arrays of 70000 images. Notice in the MATLAB folder.

#permf1 = sio.loadmat("D:\\expfracML\\data\\TrainTest75000\\permf\\permf1.mat")['permf'] # ... till 15
#permf1_1 = permf1[0, :] # ... till 15
