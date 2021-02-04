import numpy as np
import logging


#-----------------------------------------------------------------------------------------------------------------------
# How to load and see image using matplotlib
from matplotlib import image
from matplotlib import pyplot

#a1 = image.imread("D:\\mrst-2017a\\modules\\vemmech\\RESULTS\\ML\\createManyGstokes\\1.jpg")

#pyplot.imshow(a1)
#pyplot.show()



#-----------------------------------------------------------------------------------------------------------------------
# Load and concatinate permf arrays
import numpy as np
import scipy.io as sio


#permf1 = sio.loadmat("D:\\mrst-2017a\\modules\\vemmech\RESULTS\\ML\\createManyGstokes16000\\permf\\permf1.mat")['permf']
#permf2 = sio.loadmat("D:\\mrst-2017a\\modules\\vemmech\RESULTS\\ML\\createManyGstokes16000\\permf\\permf2.mat")['permf']
#permf3 = sio.loadmat("D:\\mrst-2017a\\modules\\vemmech\RESULTS\\ML\\createManyGstokes16000\\permf\\permf3.mat")['permf']
#permf4 = sio.loadmat("D:\\mrst-2017a\\modules\\vemmech\RESULTS\\ML\\createManyGstokes16000\\permf\\permf4.mat")['permf']
#permf5 = sio.loadmat("D:\\mrst-2017a\\modules\\vemmech\RESULTS\\ML\\createManyGstokes16000\\permf\\permf5.mat")['permf']


#permf1_1 = permf1[0, :]
#permf2_1 = permf2[0, 3200:]
#permf3_1 = permf3[0, 6400:]
#permf4_1 = permf4[0, 9600:]
#permf5_1 = permf5[0, 12800:]

#del permf1, permf2, permf3, permf4, permf5

#permf = np.hstack((permf1_1, permf2_1, permf3_1, permf4_1, permf5_1))


#np.savetxt("D:\\mrst-2017a\\modules\\vemmech\RESULTS\\ML\\createManyGstokes16000\\permf\\permf.csv", permf, delimiter=',')


#-----------------------------------------------------------------------------------------------------------------------
# How to plot matrix easily

# plt.imshow(your_matrix)
# plt.colorbar()


# How to set figure size
# fig.set_size_inches(10, 6)


#-----------------------------------------------------------------------------------------------------------------------
# Print formattion



print('Something %.2e' % 0.0002553) # Scientific
print('Something %s' % 0.0002553)   # String
print('Something %d' % 0.0002553)   # Integers
print('Something %f' % 0.0002553)   # Floating


# 2 placeholders %,  "John is 23 years old."
name = "John"
age = 23
print("%s is %d years old." % (name, age))

a = 1 + 1
print(a)


#-----------------------------------------------------------------------------------------------------------------------
# Logging - didn't fully understand how to use it.

logging.basicConfig(filename="logfilename.log", level=logging.INFO)
logging.info('your text goes here')


# Speficy directories
import os
os.getcwd()                  # get current working directory
os.chdir('D:\\expfracML')    # change working directory


#-----------------------------------------------------------------------------------------------------------------------
# APPENDIX

# train_Y = train_Y.reshape((imnum,))
# train_Y = train_Y.astype(int)        # convert to integer


# unique_Y = np.unique(train_Y, return_counts=False)
# num_classes = len(unique_Y)


# Check manually image and it's corresponding permeabiity, go to the raw data
# image = Image.fromarray(train_X[0, :, :])
# image.show()
# print("The first permf: " + str(train_Y[0]))


# Convert to categorical, forget about it for now.
# train_Y_label = np.array((range(1, num_classes+1)))
# train_Y_one_hot = to_categorical(train_Y, num_classes=num_classes)


# Plot model= > doesn't work
#from keras.utils import plot_model
#plot_model(model, to_file='model.png')
