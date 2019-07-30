#CONVOLUTIONAL NEURAL NETWORK
#tuned as in https://www.researchgate.net/publication/306187492_Deep_Convolutional_Neural_Networks_and_Data_Augmentation_for_Environmental_Sound_Classification

import numpy as np
import configparser
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras.utils import np_utils
from keras.backend import int_shape
from keras import regularizers
from keras import optimizers




def parse_parameters(parameters):
    parameters = parameters.split('/')
    for param in parameters:
        exec(param)


#default parameters
regularization_lambda = 0.01
reg = regularizers.l2(regularization_lambda)

kernel_size_1 = [16, 12]
kernel_size_2 = [8, 12]
kernel_size_3 = [5,7]

pool_size = [2,2]

conv1_depth = 20
conv2_depth = 28
conv3_depth = 40

drop_prob = 0.3 #dropout prob after pooling

hidden_size = 200  #num neurons of fully connected layer

time_dim = 251  #DUMMY
features_dim = 257 #DUMMY
depth = 1  #DUMMY

#modes

def PROVA_model(model_parameters='0'):

    parse_parameters(model_parameters)

    input_data = Input(shape=(time_dim, features_dim, depth)) #in tensorflow depth is the 3rd arg

    conv_1 = Convolution2D(conv1_depth, (kernel_size_1[0],kernel_size_1[1]), padding='same', activation='tanh')(input_data)
    pool_1 = MaxPooling2D(pool_size=(pool_size[0],pool_size[1]))(conv_1)

    conv_2 = Convolution2D(conv2_depth, (kernel_size_2[0],kernel_size_2[1]), padding='same', activation='tanh')(pool_1)
    pool_2 = MaxPooling2D(pool_size=(pool_size[0],pool_size[1]))(conv_2)

    conv_3 = Convolution2D(conv3_depth, (kernel_size_3[0],kernel_size_3[1]), padding='same', activation='tanh')(pool_2)
    pool_3 = MaxPooling2D(pool_size=(pool_size[0],pool_size[1]))(conv_3)

    flat = Flatten()(pool_3)

    drop_1 = Dropout(drop_prob)(flat)
    hidden = Dense(hidden_size, activation='tanh', kernel_regularizer=reg)(drop_1)

    out = Dense(8, activation='linear')(hidden)

    model = Model(inputs=input_data, outputs=out)

    return model

def OMG_model(time_dim, features_dim, model_parameters='0'):

    parse_parameters(model_parameters)

    input_data = Input(shape=(time_dim, features_dim, depth)) #in tensorflow depth is the 3rd arg

    conv_1 = Convolution2D(conv1_depth, (kernel_size_1[0],kernel_size_1[1]), padding='same', activation='tanh')(input_data)
    pool_1 = MaxPooling2D(pool_size=(pool_size[0],pool_size[1]))(conv_1)

    conv_2 = Convolution2D(conv2_depth, (kernel_size_2[0],kernel_size_2[1]), padding='same', activation='tanh')(pool_1)
    pool_2 = MaxPooling2D(pool_size=(pool_size[0],pool_size[1]))(conv_2)

    conv_3 = Convolution2D(conv3_depth, (kernel_size_3[0],kernel_size_3[1]), padding='same', activation='tanh')(pool_2)
    pool_3 = MaxPooling2D(pool_size=(pool_size[0],pool_size[1]))(conv_3)

    flat = Flatten()(pool_3)

    drop_1 = Dropout(drop_prob)(flat)
    hidden = Dense(hidden_size, activation='tanh', kernel_regularizer=reg)(drop_1)

    out = Dense(8, activation='softmax')(hidden)

    model = Model(inputs=input_data, outputs=out)

    return model




if __name__ == '__main__':
    main()
