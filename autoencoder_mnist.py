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


# Make predictions and plot
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

# You need to collect 2 zeros, 3 ones, two threes for the simple analysis


# Encoder feature extraction - notice train data for now because we focus on features not reproducibility
encoder_m = Model(inputs=img, outputs=latent_vector)
encoded_output = encoder_m.predict(x_train[:10])
encoder_m.save('encoder.h5py')


fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()
for i in range(10):
    axs[i].imshow(x_train[i])
    axs[i].set_title(str(i))

# Select data to plot clusters
a0, a1 = 2, 9  # label showing 4s - componets 0
b0, b1 = 3, 8  # label showing 1s - component 1
plt.plot(encoded_output[a0, :], encoded_output[b0, :], 'x')
plt.plot(encoded_output[a1, :], encoded_output[b1, :], 'o')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.legend(['1st two components', '2nd two components'])


#-----------------------------------------------------------------------------------------------------------------------
# Import model and work with it
import keras.models as km
encoder_m = km.load_model('encoder.h5py')


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255.0
x_test = x_test/255.0


