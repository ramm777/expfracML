# These imports is for linux to avoid multi-threading
from sys import platform
if platform == "linux" or platform == "linux2":
    import os
    os.environ["NUMEXPR_MAX_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ['OMP_NUM_THREADS'] = "1"
    os.environ['BLIS_NUM_THREADS'] = "1"
    os.environ['VECLIB_MAXIMUM_THREADS'] = "1"
    os.environ['NUMBA_NUM_THREADS'] = "1"
    os.environ['NUMEXPR_NUM_THREADS'] = "1"

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # For the saving purposes. To revert use matplotlib.use('TkAgg')

print("Current platform: " + platform)
print('Recornize stress-permeability')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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
import plotFunctions as ff2

start = time.time()


def runTraining(datapath, CNNarchitecture, imsize_x, imsize_y, batch_size, epochs, augment, losses, figures, trainedModel=[]):
    '''
    Load training data, split to validation and training and run training and save trained model
    '''

    #train_X = ff.loadPreprocessImages(datapath, imnum, coarse_imsize_x, coarse_imsize_y)
    train_X = np.load(datapath / "train_X.npy")
    train_X = train_X.astype('float32')


    train_Y = np.loadtxt(datapath / "permf.csv")
    train_Y = train_Y / np.max(train_Y)
    train_Y = train_Y.reshape(-1, 1)
    train_Y = train_Y.astype('float32')

    if CNNarchitecture == 11: # the only mixed-CNN
        train_S = np.loadtxt(datapath / "stress.csv")
        train_S = train_S / np.max(train_S)
        train_S = train_S.reshape(-1, 1)
        train_S = train_S.astype('float32')

        Train_x, Valid_x, Train_y, Valid_y, Train_s, Valid_s = train_test_split(train_X, train_Y, train_S, test_size=0.25, random_state=42)
        del train_X, train_Y, train_S
    else:
        Train_x, Valid_x, Train_y, Valid_y = train_test_split(train_X, train_Y, test_size=0.25, random_state=42)
        del train_X, train_Y


    Train_x = Train_x.reshape(-1, imsize_x, imsize_y, 1) # No idea why this is needed, try without
    Valid_x = Valid_x.reshape(-1, imsize_x, imsize_y, 1)
    Train_x = Train_x / 255.
    Valid_x = Valid_x / 255.


    # Define (create) CNN model or load pre-trained
    if trainedModel == []:
        print('Train new model')
        model = ff1.createCNNarchitecture(CNNarchitecture, imsize_x, imsize_y) # Create model and fit
        model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam())  # loss=keras.losses.mean_absolute_percentage_error
        model.summary()
    else:
        print('Continue training of pretrained model')
        print('Continue not possible at the modemnt => not finished')
        model = trainedModel
        model.summary()


    # Train_s => train stress, Valid_s => validation stress
    print('Train default (no keras data augmentation)')
    if CNNarchitecture == 11:
        result = model.fit(x=[Train_x, Train_s], y=Train_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=([Valid_x, Valid_s], Valid_y))
    else:
        result = model.fit(x=Train_x, y=Train_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(Valid_x, Valid_y))


    # Plot loss and accuracy
    epochs = np.array(result.epoch)
    res1 = "Train loss: %.2e" % result.history['loss'][-1]
    res2 = "Validation loss: %.2e" % result.history['val_loss'][-1]
    print(res1)
    print(res2)
    losses[2] = res1
    losses[3] = res2

    #------------------------------------------------------------------------
    # Plots
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

    figures.append(fig1)

    return model, result, figures, losses


def runTesting(datapath, CNNarchitecture, modelpath, imsize_x, imsize_y, losses, figures):
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

    print('Running testing... ')

    test_X = np.load(datapath / "test_X.npy" )
    test_X = test_X.astype('float32')
    test_X = test_X.reshape(-1, imsize_x, imsize_y, 1)
    test_X = test_X / 255.

    test_Y = np.loadtxt(datapath / "permf.csv")
    test_Y = test_Y.astype('float32')
    test_Y = test_Y.reshape(-1, 1)
    test_Y = test_Y / 69461.0
    print("WARNING: scaling of permf is done manually max(train_Y)")

    test_S = np.loadtxt(datapath / "stress.csv")

    if CNNarchitecture == 10 or CNNarchitecture == 11: # the only mixed-CNN
        test_S = test_S / 50000054.0
        test_S = test_S.reshape(-1, 1)
        test_S = test_S.astype('float32')
        print("WARNING: scaling of stress is done manually max(train_S)")
        print("However, max(train_S) should be == to max(test_S)")


    model = km.load_model(modelpath, custom_objects=None, compile=True)

    if CNNarchitecture == 10 or CNNarchitecture == 11:
        predicted = model.predict([test_X, test_S])
        test_loss = model.evaluate([test_X, test_S], test_Y, verbose=1)
    else:
        predicted = model.predict(test_X)
        test_loss = model.evaluate(test_X, test_Y, verbose=1)


    mape = ff.mape(test_Y, predicted)
    res3 = "Test evaluation loss: %.2e" % test_loss
    res4 = "MAPE: %.2f" % mape
    print(res3)
    print(res4)
    losses[4] = res3
    losses[5] = res4

    #------------------------------------------------------------------------
    # Plots

    # Plot Actual vs Predicted per item
    fig2 = plt.figure(2, figsize=(15, 6))
    ax1 = fig2.add_subplot(121)
    ax2 = fig2.add_subplot(122)
    ax1.plot(test_Y, predicted, 'r+', label='Predicted vs. Actual permeability')
    ax1.plot(test_Y, test_Y, label='Linear')
    ax1.set_ylabel("Predicted permeability, scaled [0,1]")
    ax1.set_xlabel("Actual (test) permeability, scaled [0,1]")
    ax1.legend()
    fig2.text(0.6, 0.52, 'Results train/test: ')
    fig2.text(0.6, 0.5, losses[:2])
    fig2.text(0.6, 0.48, losses[2])
    fig2.text(0.6, 0.46, losses[3])
    fig2.text(0.6, 0.44, losses[4])
    fig2.text(0.6, 0.42, losses[5])
    # plt.show()
    plt.close()


    # Plot Actual vs Predicted per curve
    fig3 = plt.figure(3)
    ax1 = fig3.add_subplot()
    k = 0
    n = 30
    items_no = 150 # 930 / 30 = 31 max curves
    while n <= items_no:
        line, = ax1.plot(test_S[k:n], predicted[k:n])
        ax1.plot(test_S[k:n], test_Y[k:n], color = line.get_color(), linestyle=':')
        k = k+30
        n = n+30
    ax1.set_ylabel("Permeability, scaled [0,1]")
    ax1.set_xlabel("Stress, Pa")
    ax1.legend(['Actual relationship', 'Predicted relationship'])
    plt.title('First 5 relationships')
    plt.close()

    figures.append(fig2)
    figures.append(fig3)

    return figures, losses


#-----------------------------------------------------------------------------------------------------------------------
# Inputs for training


whatToRun = "continueTraining" # Select from: "continueTraining", "singleTesting", "runBatches"


# Inputs 16000+2000 train/valid/test images (+augmentation)
path_train = Path("data/TrainTest247_processed/Augmented_centered")   # Train X, permf, stress data
path_test = Path("data/TrainTest247_processed/Test37/Centered")       # Test X and y data


imsize_x = 128
imsize_y = 128
batch_size = 16                           # Number of training examples utilized in one iteration, larger is better
epochs = 40
augment = False                           # Keras augmentation
CNNarchitecture = [11]                    # [1,4, ...]
subcases = [1]                            # [1,2,3...]
path_results = Path('results/')


#-----------------------------------------------------------------------------------------------------------------------
# Run and save results


if whatToRun == "runBatches": # Run batches of training/testing on many architectures/iterations

    for j in range(0, len(CNNarchitecture)):
        for i in subcases:

             figures = []
             losses = [float("NaN") for x in range(0, 11)]
             str1 = 'CNNarchitecture: ' + str(CNNarchitecture[j])
             str2 = 'Subcase: ' + str(i)
             losses[0] = str1
             losses[1] = str2
             print("Starting ..." + '\n' + str1 + '\n' + str2)


             # Run training
             model, result, figures, losses = runTraining(path_train, CNNarchitecture[j], imsize_x, imsize_y, batch_size, epochs, augment, losses, figures)
             modelname = "run_cnn" + str(CNNarchitecture[j]) + "_" + str(i) + ".h5py"
             model.save(modelname)

             # Save history results
             result_pd = pd.DataFrame(result.history)
             with open("run_result_cnn" + str(CNNarchitecture[j]) + "_" + str(i) + ".csv", mode='w') as file:
                 result_pd.to_csv(file)

             # Run testing
             modelpath = Path(modelname)
             figures, losses = runTesting(path_test, CNNarchitecture[j], modelpath, imsize_x, imsize_y, losses, figures)

             ff2.resultsToPDF(modelpath, path_results, figures)

             del model, figures


elif whatToRun == "continueTraining":  # Continue traning of pre-trained model and test it

    modelname = Path("run_cnn11_1.h5py")
    modelpath = Path("selected_models1/")
    model = km.load_model(modelpath / modelname, custom_objects=None, compile=True)
    print('Continue training model: ' + modelname.name)


    assert len(CNNarchitecture) == 1
    #assert int(modelname.stem[-4]) == CNNarchitecture[0]

    # Run training
    figures = []
    losses = [float("NaN") for x in range(0, 11)]
    model, result, figures, losses = runTraining(path_train, CNNarchitecture[0], imsize_x, imsize_y, batch_size, epochs, augment, losses, figures, trainedModel=model)

    modelname_new = Path(modelname.stem + "_cont" + ".h5py")
    model.save(modelname_new)

    # Run testing
    figures, losses = runTesting(path_test, CNNarchitecture[0], modelname_new, imsize_x, imsize_y, losses, figures)

    # Save results
    result_pd = pd.DataFrame(result.history)
    with open(modelname_new.stem + ".csv", mode='w') as file:
        result_pd.to_csv(file)

    ff2.resultsToPDF(modelname_new, path_results, figures)


elif whatToRun == "singleTesting":  # Run single testing

    modelname = Path("run_cnn5_20.h5py")
    modelpath = Path("selected_models1/")
    print('Single testing model: ' + modelname.name)

    assert len(CNNarchitecture) == 1
    assert int(modelname.stem[-4]) == CNNarchitecture[0]

    figures = []
    losses = [float("NaN") for x in range(0, 11)]
    figures, losses = runTesting(path_test, CNNarchitecture[0], (modelpath / modelname), imsize_x, imsize_y, losses, figures)
    ff2.resultsToPDF(modelname, path_results, figures)

else: print('Warning: select what to run')
print('Finished. Runtime, min: ',  (time.time() - start) / 60)


# Load and plot training results CSV file
#fig = ff2.loadPlotTrainResults('result_cnn2_12.csv')




