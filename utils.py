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

import os


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
    Function to load all images, convert to grey, and collect to 3D ndarray.
        Inputs:
            datapath - path to your images
            imnum - number of images to input
            coarse_imsize_x - coarse image size on x-axis
            coarse_imsize_y - coarse image size on y-axis
        Outputs:
            train_X - features data saved as numpy file
    '''
    print('Starting ... loadPreprocessImages() function')

    for i in range(1, imnum + 1):

        image_path = datapath_x / (str(i) +".jpg")

        try:
            image = Image.open(image_path)
        except:
            print("Image cannot be open, i: %s" %i)
            a = input("Press 1 to continue 0 to abort")
            if a == str(1):
                continue
            elif a == str(0):
                break


        image_resized = image.resize((coarse_imsize_x, coarse_imsize_y))
        image_grey = image_resized.convert('L')
        image_ndarray = np.asarray(image_grey)
        #image_grey.show() # plot processed image

        if i == 1:
            allimages = image_ndarray.copy()
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

            one_image = train_X[i, :, :].copy()
            one_image1 = centreMatrixNonzeros(one_image)

            new_images[i, :, :] = one_image1.copy()
            print('i: %s' % i)
            del one_image, one_image1

    return new_images


def dataAugmentation(train_X, train_Y):
    '''
    Augment original data by: 3D transposition, flipping and flipping of the transposed
    '''

    A = np.swapaxes(train_X, 1, 2)
    B = train_X[: ,: ,::-1].copy()
    C = A[: ,: ,::-1].copy()
    train_X = np.vstack((train_X, A, B, C))

    train_Y1 = train_Y.copy()
    train_Y = np.concatenate((train_Y, train_Y1, train_Y1, train_Y1))

    return train_X, train_Y

def countlines(start, lines=0, header=True, begin_start=None):
    if header:
        print('{:>10} |{:>10} | {:<20}'.format('ADDED', 'TOTAL', 'FILE'))
        print('{:->11}|{:->11}|{:->20}'.format('', '', ''))

    for thing in os.listdir(start):
        thing = os.path.join(start, thing)
        if os.path.isfile(thing):
            if thing.endswith('.py'):
                with open(thing, 'r') as f:
                    newlines = f.readlines()
                    newlines = len(newlines)
                    lines += newlines

                    if begin_start is not None:
                        reldir_of_thing = '.' + thing.replace(begin_start, '')
                    else:
                        reldir_of_thing = '.' + thing.replace(start, '')

                    print('{:>10} |{:>10} | {:<20}'.format(
                            newlines, lines, reldir_of_thing))


    for thing in os.listdir(start):
        thing = os.path.join(start, thing)
        if os.path.isdir(thing):
            lines = countlines(thing, lines, header=False, begin_start=start)

    return lines


