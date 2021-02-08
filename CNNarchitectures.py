import keras.models as km
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU




def createCNNarchitecture(no, imsize_x, imsize_y):
    '''
    Create CNN model having architecture of several layers
    no - architecture number
    '''

    if no == 1:

        model = km.Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(imsize_x, imsize_y, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(imsize_x, imsize_y, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(imsize_x, imsize_y, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(imsize_x, imsize_y, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(imsize_x, imsize_y, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Flatten())
        model.add(Dense(32))
        model.add(Dense(1))

    # -----------------------------------------------------------------------------------------------------------------------

    if no == 2:

        model = km.Sequential()

        model.add(Conv2D(32, kernel_size=(7, 7), input_shape=(imsize_x, imsize_y, 1), padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(64, kernel_size=(5, 5), input_shape=(imsize_x, imsize_y, 1), padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(imsize_x, imsize_y, 1), padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(256, kernel_size=(3, 3), input_shape=(imsize_x, imsize_y, 1), padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(imsize_x, imsize_y, 1), padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))


        model.add(Flatten())
        model.add(Dense(1))
        #model.add(LeakyReLU(alpha=0.1))
        #model.add(Dense(32, activation='linear'))
        #model.add(Dense(1))


    # -----------------------------------------------------------------------------------------------------------------------

    if no == 3:  # based on PyImage: https://www.pyimagesearch.com/2019/01/28/keras-regression-and-cnns/

        inputShape = (imsize_x, imsize_y, 1)  # image height, width and depth (no of channels, black/white = 1)
        inputs = km.Input(shape=inputShape)

        x = inputs

        x = Conv2D(16, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(16)(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Dropout(0.5)(x)

        x = Dense(4)(x)
        x = Activation("relu")(x)
        x = Dense(1)(x)

        model = km.Model(inputs, x)


    # -----------------------------------------------------------------------------------------------------------------------

    if no == 4:  # based on chemical concentration model: https://stats.stackexchange.com/questions/335836/cnn-architectures-for-regression

        inputShape = (imsize_x, imsize_y, 1)  # image height, width and depth (no of channels, black/white = 1)
        inputs = km.Input(shape=inputShape)

        x = inputs

        x = Conv2D(32, (3, 3), padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(128, (3, 3), padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
        x = Conv2D(128, (3, 3), padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
        x = Conv2D(128, (3, 3), padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Dropout(0.5)(x)
        x = Conv2D(64, (1, 1), padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Dropout(0.5)(x)
        x = Conv2D(32, (1, 1), padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Flatten()(x)
        #x = Dense(16)(x)
        x = Dense(1)(x)

        model = km.Model(inputs, x)

    else:
        print('Select CNN architecture: 1, 2 ...')


    return model