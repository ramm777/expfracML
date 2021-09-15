#-----------------------------------------------------------------------------------------------------------------------
# Scripts to collect data for ML 289 / 247 images stress-permf from the IDC+S simulation


import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import time

import utils as ff
import functions1 as ff1

start = time.time()




def prepare247Images(imnum, datapath):

    train_X = ff.loadPreprocessImages(datapath, imnum, 128, 128)

    # Enlarge images to 247*30 each = 7410
    for i in range(train_X.shape[0]):

        image = train_X[i].copy()
        duplicated = np.resize(image, (30, 128, 128))

        if i == 0:
            new_train_X = duplicated.copy()
        else:
            new_train_X = np.vstack((new_train_X, duplicated))

        del image, duplicated

    np.save('test_X.npy', new_train_X )
    print('Finsihed creating 30*x dataset of images')

#imnum = 37 # This must be the last image number, disregarding abscent images
#datapath = Path("D:\\expfracML\\data\\TrainTest247_processed\\Test37\\Raw")
#prepare247Images(imnum, datapath)


def loadmatCoarsen():

    permf = sio.loadmat("data\\TrainTest247_processed\\Test37\\rpermf.mat")['rpermf_all']
    stress= sio.loadmat("data\\TrainTest247_processed\\Test37\\rstress.mat")['rstress_all']

    # Coarsen curves of stress-perm
    permf1 = ff1.coarsenArrayLog2(permf.transpose(), 30, exp=10)
    stress1 = ff1.coarsenArrayLog2(stress.transpose(), 30, exp=10)
    permf1 = permf1.transpose()
    stress1 = stress1.transpose()

    permf2 = permf1.flatten()
    stress2 = stress1.flatten()

    # round elements
    permf2 = np.rint(permf2)
    stress2 = np.rint(stress2)

    np.savetxt('permf.csv', permf2, delimiter=',')
    np.savetxt('stress.csv', stress2, delimiter=',')


def collectStressPermf(casesnum, datapath, plot_all=False):
    """
    Collect stress and permf data from the raw matlab-idc simulations results. Notice: clean of NaNs is done manually.
    Not fully generatized function, need to clean failed subcases manually
    """

    stress_all = [np.nan for x in range(casesnum)]
    permf_all = [np.nan for y in range(casesnum)]
    no_data = []

    for caseID in range(1, casesnum+1):

        modelpath = datapath / ('case5_' + str(caseID) + '/' + 'case5_' + str(caseID) + '.mat')

        try:
            stress = sio.loadmat(modelpath)['mstresshistbc']
            permf = sio.loadmat(modelpath)['permf']
        except:
            print("No data for the case: " + str(caseID))
            print("Continue without this case")
            no_data.append(caseID)
            continue

        stress = np.reshape(stress, (stress.size, ))
        permf = np.reshape(permf, (permf.size, ))

        stress_all[caseID-1] = stress
        permf_all[caseID-1] = permf

        del permf, stress

    # Delete items from the nested list that has no data (all NaNs vectors)
    for i in no_data:
        del stress_all[i-1]
        del permf_all[i-1]
        print("INFORMATION: cleaned NaNs in curve " + str(i-1))


    # Double-check that there is no no data
    for i in range(len(permf)):
        if np.isnan(permf[i][0]):
            print('WARNING: NaNs in curve' + str(i))
            break


    # Save data
    np.save('stress.npy', stress_all)
    np.save('permf.npy', permf_all)

    #-----------------------------------------------------
    # Plot all curves in one plot
    if plot_all == True:

        fig = plt.figure(1)
        ax = fig.add_subplot()
        for j in range(casesnum):
            permf = permf_all[j].copy()
            stress = stress_all[j].copy()

            ax.plot(stress, permf)

            del permf, stress

        plt.xlabel("Stress, Pa")
        plt.ylabel("Permf, mD")
        plt.show()

    # -----------------------------------------------------

#casesnum=250
#datapath = Path("D:/mrst-2017a/modules/vemmech/RESULTS/Synthetic2/LMd_case5-2full/")
#collectStressPermf(casesnum, datapath, plot_all=True)


def checkCases(casesnum, datapath):
    """
        Check and find abscent cases from the matlab-idc simulation run
    """

    for caseID in range(1, casesnum+1):

        modelpath = datapath / ('case5_' + str(caseID) + '/' + 'case5_' + str(caseID) + '.mat')

        try:
            stress = sio.loadmat(modelpath)['mstresshistbc']
            permf = sio.loadmat(modelpath)['permf']
        except:
            print("No data for the case: " + str(caseID))
            print("Continue without this case")
            continue

    print('Finsihed checking and finding abscent cases from the matlab-idc simulation run')


#casesnum=250
#datapath = Path("D:/mrst-2017a/modules/vemmech/RESULTS/Synthetic2/LMd_case5-6full/")
#checkCases(casesnum, datapath)


def loadSaveCSV():
    """
        These are scripts and not a ready funciton to load .npy, perform padding using NaNs and save as csv for Matlab
    """

    # Load
    permf = np.load("data\TrainTest247_processed\Test37\permf.npy", allow_pickle=True)
    stress = np.load("data\TrainTest247_processed\Test37\stress.npy", allow_pickle=True)


    # Convert object ndarray (rows are not equal in size) to dictionary (rows are not equal in size)
    permf1 = dict(enumerate(permf.flatten(), 1))
    stress1 = dict(enumerate(stress.flatten(), 1))

    # Padding empty entries in the dictionary with NaNs, to make dictionary rows equal dimensions
    permf2 = pd.DataFrame.from_dict(permf1, orient='index')
    stress2 = pd.DataFrame.from_dict(stress1, orient='index')

    # Save .csv using pandas
    pd.DataFrame(permf2).to_csv("permf.csv", index=False)
    pd.DataFrame(stress2).to_csv("stress.csv", index=False)


def embedStressToImages():
    """
    This is scripts not a function to embed stress into images data
    Need to load stress and images before
    """

    new_images1 = new_images.copy()
    for i in range(len(new_images1)):
        new_images1[i, :20, :20] = test_S[i].copy()


# How to create np.array of NaNs
#stress_all = np.empty(300)
#permf_all = np.empty(300)
#stress_all[:] = np.nan
#permf_all[:] = np.nan


#-----------------------------------------------------------------------------------------------------------------------
# Modify existing images to remove velocity field, substituting it with a single non zero value

def makeBaldFractures(datapath_x, dataname):

    data = np.load(datapath_x / dataname)

    new_data = data.copy()
    new_data[new_data > 0] = 255

    np.save('test_X_bald.npy', new_data)
    print('Filename test_X_bald.npy is later renamed')


#datapath_x = Path("data\Test2000\Augmented_centered")  # Augmented, centered
#dataname = "test_X.npy"
#makeBaldFractures(datapath_x, dataname)
#print('Finished. Runtime, min: ',  (time.time() - start) / 60)

