# https://towardsdatascience.com/how-to-make-an-autoencoder-2f2d99cd5103
# https://datascience.stackexchange.com/questions/64412/how-to-extract-features-from-the-encoded-layer-of-an-autoencoder
# Try using mnist autoencoder for the GMM data


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

model = Model(inputs = img, outputs = output)
model.compile("nadam", loss = "binary_crossentropy")
result = model.fit(x_train, x_train, epochs=50, verbose=1)
model.save('decoder.h5py')


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


# Encoder feature extraction - notice train data for now
encoder = Model(inputs=img, outputs=latent_vector)
encoded_output = encoder.predict(x_train[:10])
model.save('encoder.h5py')


