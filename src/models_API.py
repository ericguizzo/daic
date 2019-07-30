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
    for param in parameters:
        exec(param)

#DEFINE HERE YOUR MODELS!!

def EXAMPLE_model(time_dim, features_dim, model_parameters='0'):
    '''
    to use this model, simply call architecture=EXAMPLE_model as a parameter
    in the UI script
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL
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
    drop_prob = 0.3
    hidden_size = 200

    #THEN CALL THIS FUNCTION TO OVERWRITE DEFAULT PARAMETERS
    #WITH PARAMETERS DEFINED IN THE UI SCRIPT
    parse_parameters(model_parameters)
    print (parameters)
    sys.exit(0)

    #FINALLY DECLARE YOUR ARCHITECTURE AND RETURN THE MODEL
    input_data = Input(shape=(time_dim, features_dim, 1))
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
