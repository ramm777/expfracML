import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

import keras
import keras.models as km
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

import matplotlib.pyplot as plt
import pandas as pd



def getScaler(datapath):
    '''
       Get sclaer for target using 16000 targets to use for testing
    '''

    train_Y = np.loadtxt(datapath / "permf.csv")
    train_Y = train_Y / 1e4  # convert to 'mD/1e4'

    # Scale from 0 to 1
    train_Y = train_Y.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(train_Y)

    return scaler


def loadPreprocessImages(datapath_x, imnum, coarse_imsize_x, coarse_imsize_y):
    '''
    Function to load all images, convert to grey, and collect to 3D ndarray ...
        Inputs:
            datapath - path to your images
            imnum - number of images to input
            coarse_imsize_x - coarse image size on x-axis
            coarse_imsize_y - coarse image size on y-axis
        Outputs:
            train_X - features data ready for the model to be trained
    '''

    for i in range(1, imnum + 1):
        image_path = datapath_x / str(i) +".jpg"
        image = Image.open(image_path)
        image_resized = image.resize((coarse_imsize_x, coarse_imsize_y))
        image_grey = image_resized.convert('L')
        image_ndarray = np.asarray(image_grey)
        #image_grey.show() # plot processed image

        if i == 1:
            allimages = image_ndarray
        else:
            allimages = np.dstack((allimages, image_ndarray))


    train_X = np.transpose(allimages)

    return train_X


def mape(y_true, y_pred):
    '''
    Calculate mean absolute percentage error
    '''
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return mape


def centreMatrixNonzeros(my_array):
    '''
    Method centers the nonzero values within 2D numpy array
    '''

    for k in range(2):
        nonempty = np.nonzero(np.any(my_array, axis=1 - k))[0]
        first, last = nonempty.min(), nonempty.max()
        shift = (my_array.shape[k] - first - last) // 2
        my_array = np.roll(my_array, shift, axis=k)

    return my_array


def runCenteringAndPlot(train_X, plot=True):
    '''
    Centre the image data stored as a np.array in train_X. plot every image to QC.
    train_X must be the shape [image_num, imsize_x, imsize_y]
    '''

    if plot==True:
        for i in range(0, train_X.shape[0]):

            one_image = train_X[i, :, :]
            fig = plt.figure(1)
            fig.set_size_inches(10, 6)
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            one_image1 = centreMatrixNonzeros(one_image)
            ax1.imshow(one_image)
            ax2.imshow(one_image1)
            plt.show()

            print('i: %s' % i)
    else:
        new_images = train_X*0
        for i in range(0, train_X.shape[0]):

            one_image = train_X[i, :, :]
            one_image1 = centreMatrixNonzeros(one_image)

            new_images[i, :, :] = one_image1
            print('i: %s' % i)
            del one_image, one_image1

    return new_images


def dataAugmentation(train_X, train_Y):
    '''
    Augment original data by: 3D transposition, flipping and flipping of the transposed
    '''

    A = np.swapaxes(train_X, 1, 2)
    B = train_X[: ,: ,::-1]
    C = A[: ,: ,::-1]
    train_X = np.vstack((train_X, A, B, C))

    train_Y1 = train_Y
    train_Y = np.concatenate((train_Y, train_Y1, train_Y1, train_Y1))

    return train_X, train_Y


def loadPlotCSV(filename):

    '''
    Function to load and plot CSV results of your training of aq machine learning model
    '''

    data = pd.read_csv(filename)
    epochs = range(0, len(data['loss']))

    fig1 = plt.figure(1, figsize=(15, 6))
    ax1 = fig1.add_subplot(121)
    ax2 = fig1.add_subplot(122)
    ax1.plot(epochs, data['loss'], 'bo', label='Training loss')
    ax1.plot(epochs, data['val_loss'], 'b', label='Validation loss')
    ax1.set_yscale('log')
    ax1.title.set_text('Semi-log plot')
    ax1.legend()
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epochs")
    # ax2.plot(epochs, result.history['loss'], 'bo', label='Training loss')
    # ax2.plot(epochs, result.history['val_loss'], 'b', label='Validation loss')
    # ax2.title.set_text('Linear plot')
    # ax2.legend()
    # ax2.set_ylabel("Loss")
    # ax2.set_xlabel("Epochs")
    fig1.text(0.6, 0.52, 'Results training: ')
    fig1.text(0.6, 0.5, str(filename))
    fig1.text(0.6, 0.48, "Train loss: %.2e" % data['loss'].iloc[-1])
    fig1.text(0.6, 0.46, "Validation loss: %.2e" % data['val_loss'].iloc[-1])
    plt.show()
    #plt.close()
