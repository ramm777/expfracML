# Author: Amanzhol Kubeyev (email: aman85work@gmail.com)
# Image segmentation exercise, where the objective of this exercise is to produce a deep convolutional neural network (DCNN) model and an
# evaluation metric
# This code consists of 3 functions, and performs data processing, deep learning training, cross-validation and testing



import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import keras
import keras.models as km
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

import pandas as pd
from pathlib import Path

from PIL import Image
import glob


#-----------------------------------------------------------------------------------------------------------------------
# Functions

def loadPreprocessImages(path_images, path_masks, coarse_size1, coarse_size2):
    '''
        Load images and keep them as ndarray. Notice the mask images are coarsened.
    '''

    # Load all images
    images_list = []
    i = 1
    for filename in glob.glob(path_images):

        image = Image.open(filename)
        images_list.append(image)
        image_ndarray = np.asarray(image)

        if i == 1:
            allimages = image_ndarray.copy()
        else:
            allimages = np.dstack((allimages, image_ndarray))

        i = i + 1
        train_X = np.transpose(allimages)


    # Load all masks
    masks_list = []
    i = 1
    for filename in glob.glob(path_masks):

        image = Image.open(filename)
        image_resized = image.resize((coarse_size1, coarse_size2))
        image_ndarray = np.asarray(image_resized)

        masks_list.append(image)

        if i == 1:
            allmasks = image_ndarray.copy()
        else:
            allmasks = np.dstack((allmasks, image_ndarray))

        i = i + 1
        train_Y = np.transpose(allmasks)

    return train_X, train_Y


def createCNNarchitecture(no, imsize_x, imsize_y):
    '''
        Create CNN model having architecture of several layers
        no - architecture number
    '''

    if no == 1:


        inputShape = (imsize_x, imsize_y, 1)  # image height, width and depth (no of channels, black/white = 1)
        inputs = km.Input(shape=inputShape)

        x = inputs

        x = Conv2D(32, (3, 3), padding="same")(x)
        x = LeakyReLU()(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(64, (3, 3), padding="same")(x)
        x = LeakyReLU()(x)
        x = BatchNormalization(axis=-1)(x)


        x = Conv2D(128, (3, 3), padding="same")(x)
        x = LeakyReLU()(x)
        x = BatchNormalization(axis=-1)(x)

        x = Conv2D(64, (3, 3), padding="same")(x)
        x = LeakyReLU()(x)
        x = BatchNormalization(axis=-1)(x)

        x = Conv2D(1, (3, 3), activation='sigmoid')(x)

        model = km.Model(inputs, x)


    else:
        print('Warning: select CNN architecture: 1, 2 ...')

    return model


def runTraining(train_X, train_Y, batch_size, epochs, losses, coarse_size1, coarse_size2):

    '''
        Load data and run training based on train/validation data and finally run testing at the end.
    '''

    # Pre-defined images size
    imsize_x = 256
    imsize_y = 256
    print('Images size are pre-defined')


    train_X = train_X.astype('float32')


    # Make target binary
    new_data = train_Y.copy()
    new_data[new_data > 0] = 1
    del train_Y
    train_Y = new_data.copy()
    del new_data


    print('Scale X data, it is from 0 to 65535, manually identified')
    scaler = np.max(train_X.flatten())
    train_X_scaled = train_X /scaler


    # Split data to 3: train, validation, test
    Train_x, Valid_x1, Train_y, Valid_y1 = train_test_split(train_X_scaled, train_Y, test_size=0.3, random_state=42)
    Test_x, Valid_x, Test_y, Valid_y = train_test_split(Valid_x1, Valid_y1, test_size=0.5, random_state=42)
    del train_X, train_Y, Valid_x1, Valid_y1


    Train_x = Train_x.reshape(-1, imsize_x, imsize_y, 1)
    Valid_x = Valid_x.reshape(-1, imsize_x, imsize_y, 1)
    Test_x  = Test_x.reshape(-1, imsize_x, imsize_y, 1)
    Train_y = Train_y.reshape(-1, coarse_size1, coarse_size2, 1)
    Valid_y = Valid_y.reshape(-1, coarse_size1, coarse_size2, 1)
    Test_y  = Test_y.reshape(-1, coarse_size1, coarse_size2, 1)


    # Create a simple CNN model
    print('Train new model')
    model = createCNNarchitecture(1, imsize_x, imsize_y)
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam())  # keras.losses.binary_crossentropy or loss='sparse_categorical_crossentropy'
    model.summary()


    # Train
    print('Train default (no keras data augmentation)')
    result = model.fit(Train_x, Train_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(Valid_x, Valid_y))


    # Print results
    epochs = np.array(result.epoch)
    res1 = "Train loss: %.2e" % result.history['loss'][-1]
    res2 = "Validation loss: %.2e" % result.history['val_loss'][-1]
    print(res1)
    print(res2)
    losses[2] = res1
    losses[3] = res2

    # Plot results
    fig1 = plt.figure(1, figsize=(15, 6))
    ax1 = fig1.add_subplot(121)
    ax2 = fig1.add_subplot(122)
    ax1.plot(epochs, result.history['loss'], 'bo', label='Training loss')
    ax1.plot(epochs, result.history['val_loss'], 'b', label='Validation loss')
    ax1.set_yscale('log')
    ax1.title.set_text('Semi-log plot')
    ax1.legend()
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epochs")
    fig1.text(0.6, 0.52, 'Results training: ')
    fig1.text(0.6, 0.5, losses[:2])
    fig1.text(0.6, 0.48, losses[2])
    fig1.text(0.6, 0.46, losses[3])
    plt.show()
    #plt.close()


    # Testing
    predicted = model.predict(Test_x)
    test_loss = model.evaluate(Test_x, Test_y, verbose=1)
    res3 = "Test evaluation loss: %.2e" % test_loss
    print(res3)
    losses[4] = res3


    # Plot actual vs. predicted one image (ideally all needs to be evaluated)
    fig2 = plt.figure(1, figsize=(15, 6))
    ax1 = fig2.add_subplot(121)
    ax1.imshow(predicted[0])
    ax2 = fig2.add_subplot(122)
    ax2.imshow(Test_y[0])
    plt.show()


    return model, result, fig1, losses


#-----------------------------------------------------------------------------------------------------------------------
# RUN


path_images = "data_seep_detection/train_images_256/*.tif"
path_masks = "data_seep_detection/train_masks_256/*.tif"
coarse_size1 = 126
coarse_size2 = 126
batch_size = 8
epochs = 10
losses = [float("NaN") for x in range(0,11)]


train_X, train_Y = loadPreprocessImages(path_images,path_masks,coarse_size1, coarse_size2)
model, result, fig1, losses = runTraining(train_X, train_Y, batch_size, epochs, losses, coarse_size1, coarse_size2)

