import keras.models as km
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

# imports for the ResNet
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, concatenate
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
#from keras.utils import plot_model
# from resnets_utils import *
from keras.initializers import glorot_uniform
import keras.backend as K

K.set_image_data_format('channels_last')
K.set_learning_phase(1)


def createCNNarchitecture(no, imsize_x, imsize_y):
    '''
    Create CNN model having architecture of several layers
    no - architecture number
    ResNet50 is no.6 (as described in https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Networks_yourself.ipynb)
    '''

    if no == 1:

        model = km.Sequential()

        model.add(Conv2D(32, kernel_size=(9, 9), input_shape=(imsize_x, imsize_y, 1), padding='same'))
        model.add(Activation('selu')) # 'relu', 'elu'
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(64, kernel_size=(7, 7), input_shape=(imsize_x, imsize_y, 1), padding='same'))
        model.add(Activation('selu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(128, kernel_size=(5, 5), input_shape=(imsize_x, imsize_y, 1), padding='same'))
        model.add(Activation('selu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(imsize_x, imsize_y, 1), padding='same'))
        model.add(Activation('selu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(imsize_x, imsize_y, 1), padding='same'))
        model.add(Activation('selu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Flatten())
        model.add(Dense(32))
        model.add(Dense(16, activation='linear'))
        model.add(Dense(1))


    # -----------------------------------------------------------------------------------------------------------------------

    elif no == 2:

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
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(32, activation='linear'))
        model.add(Dense(1))


    # -----------------------------------------------------------------------------------------------------------------------

    elif no == 3:  # based on PyImage: https://www.pyimagesearch.com/2019/01/28/keras-regression-and-cnns/

        inputShape = (imsize_x, imsize_y, 1)  # image height, width and depth (no of channels, black/white = 1)
        inputs = km.Input(shape=inputShape)

        x = inputs

        x = Conv2D(16, (3, 3), padding="same")(x)
        x = LeakyReLU()(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(32, (3, 3), padding="same")(x)
        x = LeakyReLU()(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(64, (3, 3), padding="same")(x)
        x = LeakyReLU()(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(32, (3, 3), padding="same")(x)
        x = LeakyReLU()(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(16)(x)
        x = LeakyReLU()(x)
        x = BatchNormalization(axis=-1)(x)
        x = Dropout(0.5)(x)

        x = Dense(4)(x)
        x = LeakyReLU()(x)
        x = Dense(1)(x)

        model = km.Model(inputs, x)


    # -----------------------------------------------------------------------------------------------------------------------

    elif no == 4:  # based on chemical concentration model: https://stats.stackexchange.com/questions/335836/cnn-architectures-for-regression

        inputShape = (imsize_x, imsize_y, 1)  # image height, width and depth (no of channels, black/white = 1)
        inputs = km.Input(shape=inputShape)

        x = inputs

        x = Conv2D(32, (12, 12), padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("selu")(x)
        x = Conv2D(32, (12, 12), padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("selu")(x)
        x = Conv2D(32, (12, 12), padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("selu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(64, (9, 9), padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("selu")(x)
        x = Conv2D(64, (9, 9), padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("selu")(x)
        x = Conv2D(64, (9, 9), padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("selu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(128, (7, 7), padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("selu")(x)
        x = Conv2D(128, (5, 5), padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("selu")(x)
        x = Conv2D(128, (3, 3), padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("selu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Dropout(0.5)(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("selu")(x)

        x = Dropout(0.5)(x)
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("selu")(x)

        x = Flatten()(x)
        x = Dense(16)(x)
        x = Activation("linear")(x)
        x = Dense(1)(x)

        model = km.Model(inputs, x)


    elif no == 5: # Similar to cnn1 but more standard

        model = km.Sequential()

        model.add(Conv2D(32, kernel_size=(9, 9), input_shape=(imsize_x, imsize_y, 1), padding='same'))
        model.add(Activation('selu'))  # 'relu', 'elu'
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(64, kernel_size=(7, 7), input_shape=(imsize_x, imsize_y, 1), padding='same'))
        model.add(Activation('selu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(128, kernel_size=(5, 5), input_shape=(imsize_x, imsize_y, 1), padding='same'))
        model.add(Activation('selu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(imsize_x, imsize_y, 1), padding='same'))
        model.add(Activation('selu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(imsize_x, imsize_y, 1), padding='same'))
        model.add(Activation('selu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Flatten())
        model.add(Dense(32,activation='selu'))
        model.add(Dense(16, activation='selu'))
        model.add(Dense(1, activation='selu'))


    elif no == 6: # ResNet50 as described in https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Networks_yourself.ipynb

        def identity_block(X, f, filters, stage, block):
            """
            Implementation of the identity block as defined in Figure 3

            Arguments:
            X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
            f -- integer, specifying the shape of the middle CONV's window for the main path
            filters -- python list of integers, defining the number of filters in the CONV layers of the main path
            stage -- integer, used to name the layers, depending on their position in the network
            block -- string/character, used to name the layers, depending on their position in the network

            Returns:
            X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
            """

            # defining name basis
            conv_name_base = 'res' + str(stage) + block + '_branch'
            bn_name_base = 'bn' + str(stage) + block + '_branch'

            # Retrieve Filters
            F1, F2, F3 = filters

            # Save the input value. You'll need this later to add back to the main path.
            X_shortcut = X

            # First component of main path
            X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
            X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
            X = Activation('relu')(X)

            # Second component of main path (≈3 lines)
            X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
            X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
            X = Activation('relu')(X)

            # Third component of main path (≈2 lines)
            X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
            X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

            # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
            X = Add()([X, X_shortcut])
            X = Activation('relu')(X)

            return X


        def convolutional_block(X, f, filters, stage, block, s=2):
            """
            Implementation of the convolutional block as defined in Figure 4

            Arguments:
            X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
            f -- integer, specifying the shape of the middle CONV's window for the main path
            filters -- python list of integers, defining the number of filters in the CONV layers of the main path
            stage -- integer, used to name the layers, depending on their position in the network
            block -- string/character, used to name the layers, depending on their position in the network
            s -- Integer, specifying the stride to be used

            Returns:
            X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
            """

            # defining name basis
            conv_name_base = 'res' + str(stage) + block + '_branch'
            bn_name_base = 'bn' + str(stage) + block + '_branch'

            # Retrieve Filters
            F1, F2, F3 = filters

            # Save the input value
            X_shortcut = X

            ##### MAIN PATH #####
            # First component of main path
            X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
            X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
            X = Activation('relu')(X)

            # Second component of main path (≈3 lines)
            X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
            X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
            X = Activation('relu')(X)

            # Third component of main path (≈2 lines)
            X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
            X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

            ##### SHORTCUT PATH #### (≈2 lines)
            X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
            X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

            # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
            X = Add()([X, X_shortcut])
            X = Activation('relu')(X)

            return X


        def ResNet50(input_shape=(128, 128, 1), classes=1):
            """
            Implementation of the popular ResNet50 the following architecture:
            CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
            -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

            Arguments:
            input_shape -- shape of the images of the dataset
            classes -- integer, number of classes

            Returns:
            model -- a Model() instance in Keras
            """

            # Define the input as a tensor with shape input_shape
            X_input = Input(input_shape)

            # Zero-Padding
            X = ZeroPadding2D((3, 3))(X_input)

            # Stage 1
            X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
            X = BatchNormalization(axis=3, name='bn_conv1')(X)
            X = Activation('relu')(X)
            X = MaxPooling2D((3, 3), strides=(2, 2))(X)

            # Stage 2
            X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
            X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
            X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

            # Stage 3
            X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
            X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
            X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
            X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

            # Stage 4
            X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
            X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
            X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
            X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
            X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
            X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

            # Stage 5
            X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
            X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
            X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

            # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
            X = AveragePooling2D((2, 2), name="avg_pool")(X)

            # output layer
            X = Flatten()(X)
            X = Dense(32)(X) # Aman added
            X = Dense(classes, name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

            model = Model(inputs=X_input, outputs=X, name='ResNet50')
            return model

        # Call functions to build ResNet50
        inputShape = (imsize_x, imsize_y, 1)  # image height, width and depth (no of channels, black/white = 1)
        model = ResNet50(input_shape = inputShape, classes = 1)


    elif no == 10: print('ERROR: this cnn was deleted')

    elif no == 11:

        print('Multiple inputs Keras cnn architecture (after Ahmdes review)')

        inputShape1 = (imsize_x, imsize_y, 1)  # image height, width and depth (no of channels, black/white = 1)
        inputShape2 = (1,)
        input1 = km.Input(shape=inputShape1)
        input2 = km.Input(shape=inputShape2)

        # 1st branch from first input, similar to CNN1
        # -------------------------------------------------------------------------------------------------------
        x = input1

        x = Conv2D(32, (9, 9), padding="same")(x)
        x = Activation("selu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)

        x = Conv2D(64, (7, 7), padding="same")(x)
        x = Activation("selu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)

        x = Conv2D(128, (5, 5), padding="same")(x)
        x = Activation("selu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)

        x = Conv2D(128, (3, 3), padding="same")(x)
        x = Activation("selu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)

        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("selu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)

        x = Flatten()(x)
        x = Model(inputs=input1, outputs=x)

        # the second branch opreates on the second input
        # -------------------------------------------------------------------------------------------------------

        y = km.Sequential()
        y = Dense(1, activation="selu")(input2)
        y = Model(inputs=input2, outputs=y)

        # -------------------------------------------------------------------------------------------------------

        combined = concatenate([x.output, y.output])  # combine the output of the two branches

        z = Dense(64, activation="selu")(combined)  # apply a FC layer and then a regression prediction on the combined outputs
        z = Dense(16, activation="selu")(z)
        #z = Dropout(0.5)(z)
        z = Dense(1, activation="selu")(z)

        model = Model(inputs=[x.input, y.input], outputs=z)


    else:
        print('Warning: select CNN architecture: 1, 2 ...')


    return model