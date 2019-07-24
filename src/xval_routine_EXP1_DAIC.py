import numpy as np
import sys, os
import xval_instance_GENERIC as xval

# USING EMOMODEL4 (3 CONV/POOL)
#KERNEL (16,12), (8,12), (5,7)
#CHANNELS 20 28 40
'''
parameters list:
layer_type : conv/multi
channels
stretch_factors
output_type
penalization
'''

experiments = {}

experiments[1] = ['layer_type="conv"', 'channels=1',
             'stretch_factors=[(0.8, 1.),(1.25,1.)]',
             'output_type="pooled_map"', 'stretch_penality_lambda=0.', 'fc_insize=6831',
             'training_mode="train_and_eval"', 'network_type= "1_layer"',
             'kernel_size_1=(5,5)']

experiments[2] = ['layer_type="conv"', 'channels=1',
             'stretch_factors=[(0.8, 1.),(1.25,1.)]',
             'output_type="pooled_map"', 'stretch_penality_lambda=0.', 'fc_insize=6831',
             'training_mode="train_and_eval"', 'network_type= "1_layer"','regularization_lambda = 0.003',
             'kernel_size_1=(5,5)']

experiments[3] = ['layer_type="conv"', 'channels=1',
             'stretch_factors=[(0.8, 1.),(1.25,1.)]',
             'output_type="pooled_map"', 'stretch_penality_lambda=0.', 'fc_insize=6831',
             'training_mode="train_and_eval"', 'network_type= "1_layer"','regularization_lambda = 0.01',
             'kernel_size_1=(5,5)']

experiments[4] = ['layer_type="multi"', 'channels=1',
             'stretch_factors=[(0.95, 1.),(1.053,1.)]',
             'output_type="pooled_map"', 'stretch_penality_lambda=0.', 'fc_insize=6831',
             'training_mode="train_and_eval"', 'network_type= "1_layer"',
             'kernel_size_1=(5,5)']


#run experiments
short_description = 'same as exp1, but kernel size = (5x5)'
dataset = 'daic'
num_experiment = 2
num_folds = 4
experiment_folder = '../../../copy/daic'

output_path = experiment_folder + '/experiment_' + str(num_experiment)
if not os.path.exists(output_path):
    os.makedirs(output_path)
description_path = output_path + '/experiment_' + str(num_experiment) + '_description.txt'

with open(description_path, "w") as text_file:
    text_file.write(short_description)

try:
    begin = int(sys.argv[1])
    end = int(sys.argv[2])
    gpu_ID = int(sys.argv[3])

except IndexError:
    begin = 1
    end = 34
    gpu_ID = 1


for num_run in range(begin,end+1):
    temp_params = '/'.join(experiments[num_run])
    xval.run_experiment(num_experiment,num_run,num_folds,dataset,experiment_folder,temp_params, gpu_ID)
