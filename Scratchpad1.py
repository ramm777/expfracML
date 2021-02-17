

#-----------------------------------------------------------------------------------------------------------------------
# Filename identification linux/windos

import os
from pathlib import Path

# Might have been a useful name
a = Path(__file__).stem
b = Path(__file__).name

# Another rway
c = os.path.realpath(__file__)

print(str(a))
print(str(b))
print(str(c))
print("Warning: this doesn't work on Linux for some reason")



#-----------------------------------------------------------------------------------------------------------------------
# How to save plot in linux without showing it
#import matplotlib
#matplotlib.use('Agg') # no UI backend

#import matplotlib.pyplot as plt
#import numpy as np

#t = np.arange(0.0, 2.0, 0.01)
#s = 1 + np.sin(2*np.pi*t)
#plt.plot(t, s)
#plt.title('About as simple as it gets, folks')

#plt.show()
#plt.savefig("matplotlib.png")  #savefig, don't show


#-----------------------------------------------------------------------------------------------------------------------
# Save numpy array as text
#np.savetxt(path_results / "results.txt", np.array(losses_all), delimiter=',', fmt="%s")


#-----------------------------------------------------------------------------------------------------------------------
# Example: try to augment data with Keras, by the rotation

#from keras.preprocessing.image import ImageDataGenerator
#from matplotlib import plt

## before, load your Train_x, Train_y data


## Modify it
#x = Train_x[0, :, :, :]
#y = Train_y[0, :]

#x = np.reshape(x, (1, 128,128, 1))

# Assert they have the same length
#assert len(y) == len(x)

#datagen = ImageDataGenerator(rotation_range=90)
#it = datagen.flow(x, y, batch_size= 1)
#batch = it.next()                        # From here perform again to see another iteration
#image = batch[0]
#plt.imshow(image[0,:,:,0])