from __future__ import print_function
import numpy as np
import configparser
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as utils
from torch.autograd import Variable
from multiscale_convlayer2 import MultiscaleConv2d
import sys, os
import configparser
import loadconfig


config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

#get values from config file
PRETRAINED_MODELS_FOLDER = cfg.get('training_defaults', 'pretrained_models_folder')

def parse_parameters(defaults, parameters):
    for param in parameters:
        param = param.split('=')
        item = param[0].replace(' ', '')
        value = eval(param[1].replace(' ', ''))
        defaults[item] = value
    return defaults


#DEFINE HERE YOUR MODELS!!
def EmoModel1layer(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    to use this model, simply call architecture=EXAMPLE_model as a parameter
    in the UI script
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'channels':1,
    'kernel_size_1': [10,5],
    'output_type': 'pooled_map',
    'stretch_penality_lambda': 0.,
    'stretch_factors': [],
    'hidden_size': 200,
    'fc_insize':100,
    'num_classes':1
    }

    p = parse_parameters(p, user_parameters)

    #always return model AND p!!!
    class EmoModel1layer_class(nn.Module):

        def __init__(self):
            super(EmoModel1layer_class, self).__init__()
            self.inner_state = True
            self.conv1 = nn.Conv2d(1, p['channels'], kernel_size=p['kernel_size_1'])
            self.multiscale1 = MultiscaleConv2d(1, p['channels'], kernel_size=p['kernel_size_1'], scale_factors=p['stretch_factors'],
                                               output_type=p['output_type'], stretch_penality_lambda=p['stretch_penality_lambda'])
            self.hidden = nn.Linear(p['fc_insize'], p['hidden_size'])
            self.out = nn.Linear(p['hidden_size'], p['num_classes'])

        def forward(self, X):
            training_state = self.training
            if layer_type == 'conv':
                X = F.relu(self.conv1(X))
            if layer_type == 'multi':
                X = F.relu(self.multiscale1(X, training_state))
            X = X.reshape(X.size(0), -1)
            X = F.relu(self.hidden(X))
            X = self.out(X)

            return X


    out = EmoModel1layer_class()

    return out, p
