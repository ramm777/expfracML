import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd

import utils as ff
import functions1 as ff1


#-----------------------------------------------------------------------------------------------------------------------
# Scripts to collect data for ML 300 imagfes stress-permf



def collectStressPermf():
    """
    Collect stress and permf data from 300 simulations. Notice: clean of NaNs is done manually.
    Not fully generatized function, but more hard-coded
    """

    casesum = 250

    stress_all = [np.nan for x in range(casesum)]
    permf_all = [np.nan for y in range(casesum)]

    for caseID in range(1, casesum+1):

        datapath = Path("D:/mrst-2017a/modules/vemmech/RESULTS/Synthetic2/LMd_case5full/")
        modelpath = datapath / ('case5_' + str(caseID) + '/' + 'case5_' + str(caseID) + '.mat')

        try:
            stress = sio.loadmat(modelpath)['mstresshistbc']
            permf = sio.loadmat(modelpath)['permf']
        except:
            print("No data for the case: " + str(caseID))
            print("Continue without this case")
            continue

        stress = np.reshape(stress, (stress.size, ))
        permf = np.reshape(permf, (permf.size, ))

        stress_all[caseID-1] = stress
        permf_all[caseID-1] = permf

        print("WARNING: Cleaned NaNs manually, but this function saves unclened data")
        print("WARNING: perform cleaning of NaNs")

        np.save('stress.npy', stress_all)
        np.save('permf.npy', permf_all)


def loadSaveCSV():
    """
        These are scripts and not a ready funciton to load .npy, perfomr padding using NaNs and save as csv for Matlab
    """

    # Load
    permf = np.load("D:\expfracML\data\TrainTest289\permf.npy", allow_pickle=True)
    stress = np.load("D:\expfracML\data\TrainTest289\stress.npy", allow_pickle=True)


    # Convert object ndarray (rows are not equal in size) to dictionary (rows are not equal in size)
    permf1 = dict(enumerate(permf.flatten(), 1))
    stress1 = dict(enumerate(stress.flatten(), 1))

    # Padding empty entries in the dictionary with NaNs, to make dictionary rows equal dimensions
    permf2 = pd.DataFrame.from_dict(permf1, orient='index')
    stress2 = pd.DataFrame.from_dict(stress1, orient='index')

    # Save .csv using pandas
    pd.DataFrame(permf2).to_csv("output.csv", index=False)
    pd.DataFrame(stress2).to_csv("output.csv", index=False)



# Coarsen curves of stress-perm
#stress1 = ff1.coarsenArrayLog2(stress, 30, exp=10)
#permf1 = ff1.coarsenArrayLog2(permf, 30, exp=10)



# How to create np.array of NaNs
#stress_all = np.empty(300)
#permf_all = np.empty(300)
#stress_all[:] = np.nan
#permf_all[:] = np.nan


#-----------------------------------------------------------------------------------------------------------------------
# Modify existing images to remove velocity field, substituting it with a single non zero value


#datapath_x = Path("data\Test2000\Augmented_centered")  # Augmented, centered
#dataname = "test_X.npy"
#makeBaldFractures(datapath_x, dataname)


def makeBaldFractures(datapath_x, dataname):

    data = np.load(datapath_x / dataname)

    new_data = data.copy()
    new_data[new_data > 0] = 255

    np.save('test_X_bald.npy', new_data)
    print('Filename test_X_bald.npy is later renamed')


#-----------------------------------------------------------------------------------------------------------------------
# Data engineering


# Run data augmentation
#datapath_y = "D:\\expfracPython\\data_recognizeStokes\\Train\\Augmented\\permf.csv"     # Augmented
#datapath_x = "D:\\expfracPython\\data_recognizeStokes\\Train\\Augmented\\train_X.npy"   # Augmented
#train_Y = np.loadtxt(datapath_y)
#train_X = np.load(datapath_x)
#train_X, train_Y = ff.dataAugmentation(train_X, train_Y)
#np.save('train_X.npy', train_X)
#np.save('train_Y.npy', train_Y)


# Run centering scripts
#train_X = np.load("D:\\expfracPython\\data_recognizeStokes\\Augmented\\train_X.npy")  # Augmented
#new_images = ff.runCenteringAndPlot(train_X, plot=False)
#np.save('new_images.npy', new_images)


# Run data processing, augmentation and centering
#imnum2 = 75110
#datapath_x = Path("D:\\expfracML\\data\\TrainTest75110\\")

#train_X = ff.loadPreprocessImages(datapath_x, imnum2, 128, 128)
#np.save('train_X.npy', train_X)
#print('Finished loading and processing data')


#datapath_y = Path("D:\\expfracML\\data\\TrainTest75110\\permf\\permf.csv")     # Augmented
#train_Y = np.loadtxt(datapath_y)


#train_X, train_Y = ff.dataAugmentation(train_X, train_Y)
#new_images = ff.runCenteringAndPlot(train_X, plot=False)
#np.save('train_X.npy', new_images)
#np.savetxt("train_Y.csv", train_Y, delimiter=',')


#-----------------------------------------------------------------------------------------------------------------------
# Load and concatinate permf arrays


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


#np.savetxt("permf.csv", permf, delimiter=',')