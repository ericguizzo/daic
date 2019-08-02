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
import models_API as choose_model


def c():
    s = 'a=2'
    exec(s)
    print (locals()['a'])

c()
