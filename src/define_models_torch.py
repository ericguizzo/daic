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
def CNN_1conv(time_dim, features_dim, user_parameters=['niente = 0']):
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
    'num_classes':8,
    'dropout': False,
    'drop_prob': 0.4
    }

    p = parse_parameters(p, user_parameters)

    print ('CULO')
    print (p)

    #always return model AND p!!!
    class CNN_1conv_class(nn.Module):

        def __init__(self):
            super(CNN_1conv_class, self).__init__()
            self.layer_type = p['layer_type']
            self.inner_state = True
            if self.layer_type == 'conv':
                self.conv1 = nn.Conv2d(1, p['channels'], kernel_size=p['kernel_size_1'])
            if self.layer_type == 'multi':
                self.multiscale1 = MultiscaleConv2d(1, p['channels'], kernel_size=p['kernel_size_1'], scale_factors=p['stretch_factors'],
                                            output_type=p['output_type'], stretch_penality_lambda=p['stretch_penality_lambda'])
            self.hidden = nn.Linear(p['fc_insize'], p['hidden_size'])
            self.out = nn.Linear(p['hidden_size'], p['num_classes'])
            self.dropout = p['dropout']
            self.drop_prob = p['drop_prob']

        def forward(self, X):
            training_state = self.training
            if self.layer_type == 'conv':
                X = F.relu(self.conv1(X))
            if self.layer_type == 'multi':
                X = F.relu(self.multiscale1(X, training_state))
            X = X.reshape(X.size(0), -1)
            if self.dropout:
                X = F.dropout2d(X, self.drop_prob)
            X = F.relu(self.hidden(X))
            X = self.out(X)

            return X


    out = CNN_1conv_class()

    return out, p


def CNN_2conv(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    to use this model, simply call architecture=EXAMPLE_model as a parameter
    in the UI script
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'channels':10,
    'kernel_size_1': [10,5],
    'output_type': 'pooled_map',
    'stretch_penality_lambda': 0.,
    'stretch_factors': [],
    'hidden_size': 200,
    'fc_insize':100,
    'num_classes':8,
    'dropout': False,
    'drop_prob': 0.4,
    'pool_size': [2,2]
    }

    p = parse_parameters(p, user_parameters)

    #always return model AND p!!!
    class CNN_2conv_class(nn.Module):
        def __init__(self):
            super(CNN_2conv_class, self).__init__()
            self.layer_type = p['layer_type']
            self.inner_state = True
            if p['layer_type'] == 'conv':
                self.conv1 = nn.Conv2d(1, p['channels'], kernel_size=p['kernel_size_1'])
                self.conv2 = nn.Conv2d(p['channels'], p['channels'], kernel_size=p['kernel_size_1'])
            if p['layer_type'] == 'multi':
                self.multiscale1 = MultiscaleConv2d(1, p['channels'], kernel_size=p['kernel_size_1'], scale_factors=p['stretch_factors'],
                                                output_type=p['output_type'], stretch_penality_lambda= p['stretch_penality_lambda'])
                self.multiscale2 = MultiscaleConv2d(p['channels'], p['channels'], kernel_size=p['kernel_size_1'], scale_factors=p['stretch_factors'],
                                                output_type=p['output_type'], stretch_penality_lambda= p['stretch_penality_lambda'])
            self.pool = nn.MaxPool2d(p['pool_size'][0], p['pool_size'][1])
            self.hidden = nn.Linear(p['fc_insize'], p['hidden_size'])
            self.out = nn.Linear(p['hidden_size'], p['num_classes'])
            self.dropout = p['dropout']
            self.drop_prob = p['drop_prob']

        def forward(self, X):
            training_state = self.training
            if self.layer_type == 'conv':
                X = F.relu(self.conv1(X))
                X = self.pool(X)
                X = F.relu(self.conv2(X))
            if self.layer_type == 'multi':
                X = F.relu(self.multiscale1(X, training_state))
                X = self.pool(X)
                X = F.relu(self.multiscale2(X, training_state))
            X = X.reshape(X.size(0), -1)
            if self.dropout:
                X = F.dropout2d(X, self.drop_prob)
            X = F.relu(self.hidden(X))
            X = self.out(X)

            return X


    out = CNN_2conv_class()

    return out, p
