# https://towardsdatascience.com/how-to-make-an-autoencoder-2f2d99cd5103
# https://datascience.stackexchange.com/questions/64412/how-to-extract-features-from-the-encoded-layer-of-an-autoencoder
# Try using mnist autoencoder for the GMM data
# TODO transfer this to the separate project, taking venv from here.


from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input, Flatten, Reshape, LeakyReLU as LR, Activation, Dropout
from tensorflow.keras.models import Model, Sequential
from matplotlib import pyplot as plt
from IPython import display # If using IPython, Colab or Jupyter
import numpy as np




#-----------------------------------------------------------------------------------------------------------------------
# Construct and train autoencoder

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255.0
x_test = x_test/255.0

# Plot image data from x_train
plt.imshow(x_train[0], cmap = "gray")
plt.show()


LATENT_SIZE = 32
encoder = Sequential([
    Flatten(input_shape = (28, 28)),
    Dense(512),
    LR(),
    Dropout(0.5),
    Dense(256),
    LR(),
    Dropout(0.5),
    Dense(128),
    LR(),
    Dropout(0.5),
    Dense(64),
    LR(),
    Dropout(0.5),
    Dense(LATENT_SIZE),
    LR()
])

decoder = Sequential([
    Dense(64, input_shape = (LATENT_SIZE,)),
    LR(),
    Dropout(0.5),
    Dense(128),
    LR(),
    Dropout(0.5),
    Dense(256),
    LR(),
    Dropout(0.5),
    Dense(512),
    LR(),
    Dropout(0.5),
    Dense(784),
    Activation("sigmoid"),
    Reshape((28, 28))
])

img = Input(shape = (28, 28))
latent_vector = encoder(img)
output = decoder(latent_vector)

decoder_m = Model(inputs = img, outputs = output)
decoder_m.compile("nadam", loss ="binary_crossentropy")
result = decoder_m.fit(x_train, x_train, epochs=50, verbose=1)
decoder_m.save('decoder.h5py')


# Make predictions of selected images data and plot
n_img = 6
predict = model.predict(x_test[:n_img])
for i in range(n_img):
    fig2 = plt.figure(figsize=(6,6))
    ax1 = fig2.add_subplot(1,2,1)
    ax2 = fig2.add_subplot(1,2,2)
    ax1.imshow(x_test[i])
    ax2.imshow(predict[i])
    ax1.title.set_text('Original x_test')
    ax2.title.set_text('Predicted by autoencoder')
    plt.show()


# Encoder feature extraction - notice train data for now because we focus on features not reproducibility
encoder_m = Model(inputs=img, outputs=latent_vector)
encoder_m.save('encoder.h5py')


#-----------------------------------------------------------------------------------------------------------------------
# Import model and work with it. This can run independantly of above

import keras.models as km
from sklearn import mixture
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

import functions as ff

# Select manually:
n_img = 'all'         #  'all' or 10
n_components =  10  # 10 in all, 7 in 10 first images
covariance_type = 'full'


encoder_m = km.load_model('selected_models/encoder.h5py')

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255.0
x_test = x_test/255.0


if n_img == 'all':
    x = x_train.copy()
    y = y_train.copy()
else:
    x = x_train[:n_img].copy()
    y = y_train[:n_img].copy()
img_fvs = encoder_m.predict(x) # image featureVectors
del x_train, y_train


plot10images = False
if plot10images == True:
    # Plot the first 10 images of x_train
    fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = axs.ravel()
    for i in range((len(x))):
        axs[i].imshow(x[i])
        axs[i].set_title(str(i))

    # Select data to plot clusters
    a0, a1 = 2, 9  # label showing '4' - componets 1
    b0, b1 = 3, 8  # label showing '1' - component 2
    plt.plot(img_fvs[a0, :], img_fvs[b0, :], 'x')
    plt.plot(img_fvs[a1, :], img_fvs[b1, :], 'o')
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.legend(['1st two components', '2nd two components'])


#-----------------------------------------------------------------------------------------------------------------------
# GMM training.


gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=covariance_type)
gmm.fit(img_fvs)
proba = gmm.predict_proba(img_fvs)   # predict probabilities for each row.
print('GMM converged: ', gmm.converged_)
# X_pred = clf.fit_predict(img_fvs)  # predict labels per row


y_onehot = pd.get_dummies(y)
labels = y_onehot.columns.values

# Topic similarity
topic_similarity = distance.cdist(np.array(y_onehot.T), proba.T)
# topic_matching = topic_similarity.argmin(axis=1)              # Find non-unique topic matching
topic_matching = ff.findUniqueTopicMatching(topic_similarity)      # Find unique topic matching
reordered_onehot = proba[:, topic_matching]
reordered_onehot = pd.DataFrame(reordered_onehot, columns=labels, dtype=int)
y_hat = np.array(reordered_onehot.idxmax(axis=1))


# Plot min distance matrix
plt.figure(figsize=(6,6))
plt.imshow(topic_similarity)
plt.title("Minimum distance matrix", fontsize=15)
plt.colorbar(shrink=1)
plt.xlabel('inferred topics')
plt.ylabel('original topics')
plt.show()


# Evaluation metrics (turn to classification)
accuracy = accuracy_score(y, y_hat)
print('Accuracy classification: %0.3f' % accuracy)


# To visualize only
#y = y[:, None]
#y_hat = y_hat[:, None]


# Confusion matrix calculate and plot
cm = confusion_matrix(y, y_hat, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation='vertical') # 'vertical'
plt.rcParams["figure.figsize"] = (15, 10)  # Changing the default parameters
plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"] # Restoring the default parameters
plt.show()
