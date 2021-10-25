import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import utils as ff
import plotFunctions as vis



# Load and plot training results CSV file
vis.setPlotStyles()
datapath = Path("selected_models_paper/Not_bald")
fig = vis.loadPlotTrainResults(datapath / 'result_cnn6_11.csv')



def dummy():

    path_results = Path('results/')

    fig1 = plt.figure(1)
    plt.plot([1, 2, 3], [10, 20, 30])
    plt.close()
    fig2 = plt.figure(1)
    plt.plot([10, 20, 30], [100, 200, 300], 'rx')
    plt.close()

    pdfname = "1.pdf"
    pdf = PdfPages(path_results / pdfname)

    figures = []
    figures.append(fig1)
    figures.append(fig2)

    for i in range(len(figures)):
        pdf.savefig(figures[i])
    pdf.close()



def saveMultipleCurves():
    """
        Example of multiple curves on one plot in console mode (not in console is much easier by the way)
    """
    fig = plt.figure(1)
    ax = fig.add_subplot()
    ax.plot([1,2,3,4,5], [1,2,3,4,5])
    ax.plot([10,20,30,40,50], [1,2,3,4,5])



#dummy()
#ff.countlines(r'D:\expfracML')



#-----------------------------------------------------------------------------------------------------------------------
# How to load and see image using matplotlib
#from matplotlib import image
#from matplotlib import pyplot

#a1 = image.imread("D:\\mrst-2017a\\modules\\vemmech\\RESULTS\\ML\\createManyGstokes\\1.jpg")

#pyplot.imshow(a1)
#pyplot.show()


#-----------------------------------------------------------------------------------------------------------------------
# How to plot matrix easily

# plt.imshow(your_matrix)
# plt.colorbar()


# How to set figure size
# fig.set_size_inches(10, 6)

#-----------------------------------------------------------------------------------------------------------------------
# Plot testing old

# Plot vs each item
# fig2 = plt.figure(2, figsize=(15, 8))
# ax1 = fig2.add_subplot(1, 2, 1)
# ax2 = fig2.add_subplot(1, 2, 2)
# ax1.plot(range(implot), predicted[:implot], 'r+', label='Predicted perm')
# ax1.plot(range(implot), test_Y_scaled[:implot], 'bo', label='Actual perm')
# ax1.title.set_text('Predicted vs Actual, scaled')
# ax1.legend()
# ax1.set_ylabel("Permeability, scaled [0 to 1]")
# ax1.set_xlabel("Item no.")
# ax2.plot(range(implot), predicted_unscaled[:implot], 'r+', label='Predicted perm')
# ax2.plot(range(implot), test_Y[:implot], 'bo', label='Actual perm')
# ax2.title.set_text('Predicted vs Actual, mD/1e4')
# ax2.legend()
# ax2.set_ylabel("Permeability, mD/1e4")
# ax2.set_xlabel("Item no.")
# # plt.show()
# plt.close()


def exampleOfPrint():

    """
    Print formats example
    """

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

