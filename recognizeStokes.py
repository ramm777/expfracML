import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # For the saving purposes. To revert use matplotlib.use('TkAgg')
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


start = time.time()


def runTraining(datapath, CNNarchitecture, imsize_x, imsize_y, batch_size, epochs, scaler, augment,losses, trainedModel=[]):
    '''
    Load training data, split to validation and training and run training and save trained model
    '''

    #train_X = ff.loadPreprocessImages(datapath, imnum, coarse_imsize_x, coarse_imsize_y)
    train_X = np.load(datapath / "train_X.npy")
    train_X = train_X.astype('float32')


    train_Y = np.loadtxt(datapath / "permf.csv")
    train_Y = train_Y / 1e4  # convert to 'mD/1e4'
    train_Y = train_Y.reshape(-1, 1)
    train_Y_scaled = scaler.transform(train_Y)
    # data_unscaled = scaler.inverse_transform(data_scaled) # to unscale the data back


    Train_x, Valid_x, Train_y, Valid_y = train_test_split(train_X, train_Y_scaled, test_size=0.25, random_state=42)
    del train_X, train_Y


    Train_x = Train_x.reshape(-1, imsize_x, imsize_y, 1)
    Valid_x = Valid_x.reshape(-1, imsize_x, imsize_y, 1)
    Train_x = Train_x / 255.
    Valid_x = Valid_x / 255.


    # Define (create) CNN model or load pre-trained
    if trainedModel == []:
        print('Train new model')
        model = ff1.createCNNarchitecture(CNNarchitecture, imsize_x, imsize_y) # Create model and fit
        model.compile(loss=keras.losses.mean_squared_error, optimizer='adam')  # loss=keras.losses.mean_absolute_percentage_error
        model.summary()
    else:
        print('Continue training of pretrained model')
        model = trainedModel
        model.summary()


    # Train either by default or by using augmentation in Keras
    if augment == False:
        print('Train default (no keras data augmentation)')
        result = model.fit(Train_x, Train_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(Valid_x, Valid_y))
    else:
        print('Train with keras data augmentation')
        datagen = ImageDataGenerator(rotation_range=90)
        it = datagen.flow(Train_x, Train_y, batch_size = batch_size)
        result = model.fit(it, validation_data=(Valid_x, Valid_y), steps_per_epoch=np.ceil(len(Train_y)/batch_size), epochs=epochs)


    # Plot loss and accuracy
    epochs = np.array(result.epoch)
    res1 = "Train loss: %.2e" % result.history['loss'][-1]
    res2 = "Validation loss: %.2e" % result.history['val_loss'][-1]
    print(res1)
    print(res2)
    losses[2] = res1
    losses[3] = res2

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
    #ax2.plot(epochs, result.history['loss'], 'bo', label='Training loss')
    #ax2.plot(epochs, result.history['val_loss'], 'b', label='Validation loss')
    #ax2.title.set_text('Linear plot')
    #ax2.legend()
    #ax2.set_ylabel("Loss")
    #ax2.set_xlabel("Epochs")
    fig1.text(0.6, 0.52, 'Results training: ')
    fig1.text(0.6, 0.5, losses[:2])
    fig1.text(0.6, 0.48, losses[2])
    fig1.text(0.6, 0.46, losses[3])
    #plt.show()
    plt.close()

    return model, result, fig1, losses


def runTesting(datapath, modelpath, imsize_x, imsize_y, scaler, losses):
    '''
    Load saved ML model and testing data and run evaluation and prediction
        Inputs:
            datapath - path to your images (as jpg or matrix) and to your targets 'y'
            implot - number of images to plot, as plot can be too big
            imsize_x - image size on x-axis
            imsize_y - image size on y-axis
        Outputs:
            Prints test evaluation metrics and plots

    '''

    test_X = np.load(datapath / "test_X.npy" )
    print('Running testing... ')


    test_X = test_X.astype('float32')
    test_X = test_X.reshape(-1, imsize_x, imsize_y, 1)
    test_X = test_X / 255.


    test_Y = np.loadtxt(datapath / "permf.csv")
    test_Y = test_Y / 1e4  # convert to 'mD/1e4'
    test_Y = test_Y.reshape(-1, 1)
    test_Y_scaled = scaler.transform(test_Y)


    model = km.load_model(modelpath, custom_objects=None, compile=True)


    predicted = model.predict(test_X)
    predicted_unscaled = scaler.inverse_transform(predicted)  # to unscale the data back


    test_loss = model.evaluate(test_X, test_Y_scaled, verbose=1)
    mape = ff.mape(test_Y, predicted_unscaled)
    res3 = "Test evaluation loss: %.2e" % test_loss
    res4 = "MAPE: %.2f" % mape
    print(res3)
    print(res4)
    losses[4] = res3
    losses[5] = res4


    # Convert units to Darcy
    test_Y = test_Y*10
    predicted_unscaled = predicted_unscaled*10

    # Plot
    fig2 = plt.figure(2, figsize=(15, 6))
    ax1 = fig2.add_subplot(121)
    ax2 = fig2.add_subplot(122)
    ax1.plot(test_Y, predicted_unscaled, 'r+', label='Predicted vs. Actual permeability')
    ax1.plot(test_Y, test_Y, label='Linear')
    ax1.set_ylabel("Predicted permeability, D")
    ax1.set_xlabel("Actual (test) permeability, D")
    ax1.legend(frameon=False)
    fig2.text(0.6, 0.52, 'Results train/test: ')
    fig2.text(0.6, 0.5, losses[:2])
    fig2.text(0.6, 0.48, losses[2])
    fig2.text(0.6, 0.46, losses[3])
    fig2.text(0.6, 0.44, losses[4])
    fig2.text(0.6, 0.42, losses[5])
    # plt.show()
    plt.close()

    return fig2, losses

#-----------------------------------------------------------------------------------------------------------------------
# Inputs for training


if __name__ == '__main__':

    whatToRun = "singleTesting"  # Select from: "continueTraining", "singleTesting", "runBatches"

    # Inputs 16000+2000 train/valid/test images (+augmentation)
    path_train = Path("data/Train/Augmented_centered/")   # Train X and y data
    path_test = Path("data/Test2000/Augmented_centered/") # Test X and y data, original
    #path_test = Path("data/Carmel_nodepth_test2000/Augmented_centered/Bald/")  # Test X and y data, No depth Carmel sample
    path_traintest = path_train


    imsize_x = 128
    imsize_y = 128
    batch_size = 16                                         # Number of training examples utilized in one iteration, larger is better
    epochs = 5
    augment = True                                          # Keras augmentation
    CNNarchitecture = [6]                                   # [1,4, ...]
    subcases = [1]                                          # [1,2,3...]


    scaler = ff.getScaler(path_traintest)  # Scale from 0 to 1
    path_results = Path('results/')
    vis.setPlotStyles()

    #-----------------------------------------------------------------------------------------------------------------------
    # Run and save results


    if whatToRun == "runBatches": # Run batches of training/testing on many architectures/iterations

        for j in range(0, len(CNNarchitecture)):
            for i in subcases:

                 losses = [float("NaN") for x in range(0,11)]
                 str1 = 'CNNarchitecture: ' + str(CNNarchitecture[j])
                 str2 = 'Subcase: ' + str(i)
                 losses[0] = str1
                 losses[1] = str2
                 print("Starting ..." + '\n' + str1 + '\n' + str2)


                 # Run training
                 model, result, fig1, losses = runTraining(path_train, CNNarchitecture[j], imsize_x, imsize_y, batch_size, epochs, scaler, augment, losses)
                 modelname = "model_cnn" + str(CNNarchitecture[j]) + "_" + str(i) + ".h5py"
                 model.save(modelname)

                 # Save results
                 result_pd = pd.DataFrame(result.history)
                 with open("result_cnn" + str(CNNarchitecture[j]) + "_" + str(i) + ".csv", mode='w') as file:
                     result_pd.to_csv(file)

                 # Run testing
                 modelpath = modelname
                 fig2, losses = runTesting(path_test, modelpath, imsize_x, imsize_y, scaler, losses)

                 # Save plots to pdf
                 pdfname = "results" + str(CNNarchitecture[j]) + "_" + str(i) + ".pdf"
                 pdf = PdfPages(path_results / pdfname)
                 pdf.savefig(fig1)
                 pdf.savefig(fig2)
                 pdf.close()

                 del fig1, fig2, model


    elif whatToRun == "continueTraining":  # Continue traning of pre-trained model and test it

        modelname = "model_cnn6_4.h5py"
        modelpath = Path("selected_models/")
        model = km.load_model(modelpath / modelname, custom_objects=None, compile=True)

        # Run training
        losses = [float("NaN") for x in range(0, 11)]
        model, result, fig1, losses = runTraining(path_train, CNNarchitecture, imsize_x, imsize_y, batch_size, epochs, scaler, augment, losses, trainedModel=model)
        modelname_new = modelname[:-5] + "_cont" + ".h5py"
        model.save(modelname_new)

        # Run testing
        fig2, losses = runTesting(path_test, modelname_new, imsize_x, imsize_y, scaler, losses=losses)

        # Save results
        result_pd = pd.DataFrame(result.history)
        with open(modelname_new[:-5] + ".csv", mode='w') as file:
            result_pd.to_csv(file)

        pdfname = modelname_new[:-5] + ".pdf"
        pdf = PdfPages(path_results / pdfname)  # Save results to pdf
        pdf.savefig(fig1)
        pdf.savefig(fig2)
        pdf.close()


    elif whatToRun == "singleTesting":

        modelname = "model_cnn2_12.h5py"
        modelpath = Path("selected_models_paper/Bald/")

        losses = [float("NaN") for x in range(0, 11)]
        fig2, losses = runTesting(path_test, (modelpath / modelname), imsize_x, imsize_y, scaler, losses)

        pdfname = modelname[:-5] + ".pdf"
        pdf = PdfPages(path_results / pdfname )  # Save results to pdf
        pdf.savefig(fig2)
        pdf.close()

    else: print('Warning: select what to run')

    print('Finished. Runtime, min: ',  (time.time() - start) / 60)


#-----------------------------------------------------------------------------------------------------------------------
# APPENDIX

# These imports is for linux to avoid multi-threading
#from sys import platform
#if platform == "linux":
#    import os
#    os.environ["NUMEXPR_MAX_THREADS"] = "1"
#    os.environ["OPENBLAS_NUM_THREADS"] = "1"
#    os.environ["NUMEXPR_NUM_THREADS"] = "1"
#    os.environ["MKL_NUM_THREADS"] = "1"
#    os.environ['OMP_NUM_THREADS'] = "1"
#    os.environ['BLIS_NUM_THREADS'] = "1"
#    os.environ['VECLIB_MAXIMUM_THREADS'] = "1"
#    os.environ['NUMBA_NUM_THREADS'] = "1"
#    os.environ['NUMEXPR_NUM_THREADS'] = "1"