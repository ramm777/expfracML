

from sys import platform
if platform == "linux":
    print('Works fine on linux')


def visualizeModelSchamatically(no, imsize_x, imsize_y):
    '''
    Create a graph plot of your deep learning model. Works on linux only.
    '''

    import CNNarchitectures as ff1
    from keras.utils.vis_utils import plot_model

    model = ff1.createCNNarchitecture(no, imsize_x, imsize_y)
    plot_model(model, to_file='model_schematics.png', show_shapes=True, show_layer_names=True)

    print("Warning: this doesn't work on Windows, works on Linux only")
    print("Warning: for windows see this fix: https://stackoverflow.com/questions/47605558/importerror-failed-to-import-pydot-you-must-install-pydot-and-graphviz-for-py")


def identifyFilename():
    '''
    Funciton to identify a current filename identification windows
    '''

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


def someScripts():
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