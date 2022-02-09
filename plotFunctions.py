import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import pandas as pd

def setPlotStyles():
    """
    Set plot styles for your figure: font of text, axes legend etc...
    """

    # Restore default style/fonts in matplotlib
    # matplotlib.pyplot.rcdefaults()

    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # plt.rc('font', family='serif')

    return

def resultsToPDF(modelname, to_folder, figures):
    """
        Save all figures to .pdf file into the folder specified by "to_folder" path
    """

    pdfname = modelname.with_suffix(".pdf")
    pdf = PdfPages(to_folder / pdfname)

    for i in range(len(figures)):
        pdf.savefig(figures[i])
    pdf.close()


def loadPlotTrainResults(filename):

    '''
    Function to load and plot training results of your ML model saved as .csv file
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
    ax1.set_ylabel("Loss, dimensionless")
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

    return fig1


# Subplots
#fig2 = plt.figure(figsize=(6,6))
#ax1 = fig2.add_subplot(1,2,1)
#ax2 = fig2.add_subplot(1,2,2)
#ax1.imshow(Z)
#ax2.imshow(doc_topic_dist)
#ax1.title.set_text('Z (fitting results)')
#ax2.title.set_text('doc_topic')

# Subplots - quick way of plotting many subplots on one figure
#fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
#fig.subplots_adjust(hspace = .5, wspace=.001)
#axs = axs.ravel()
#for i in range(10):
#    axs[i].imshow(x_train[i])
#    axs[i].set_title(str(i))