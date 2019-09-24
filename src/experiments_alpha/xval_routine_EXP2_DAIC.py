import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import xval_instance_GENERIC_API as xval

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


#10 CHANNELS, 1LAYER

experiments[1] = ['layer_type="conv"', 'channels1_daic=10', 'channels2_daic=10',
             'kernel_size_1_daic = [30,5]', 'kernel_size_2_daic = [3,257]',
             'stretch_factors=[(0.8, 1.),(1.25,1.)]',
             'output_type="pooled_map"', 'stretch_penality_lambda=0.', 'fc_insize=16779600',
             'training_mode="train_and_eval"', 'network_type= "2_layer"']

experiments[2] = ['layer_type="conv"', 'channels1_daic=10', 'channels2_daic=10',
             'kernel_size_1_daic = [30,5]', 'kernel_size_2_daic = [3,257]',
             'stretch_factors=[(0.8, 1.),(1.25,1.)]',
             'output_type="pooled_map"', 'stretch_penality_lambda=0.', 'fc_insize=16779600',
             'training_mode="train_and_eval"', 'network_type= "2_layer"','regularization_lambda = 0.003']

experiments[3] = ['layer_type="conv"', 'channels1_daic=10', 'channels2_daic=10',
             'kernel_size_1_daic = [30,5]', 'kernel_size_2_daic = [3,257]',
             'stretch_factors=[(0.8, 1.),(1.25,1.)]',
             'output_type="pooled_map"', 'stretch_penality_lambda=0.', 'fc_insize=16779600',
             'training_mode="train_and_eval"', 'network_type= "2_layer"','regularization_lambda = 0.01']

experiments[4] = ['layer_type="multi"', 'channels1_daic=10', 'channels2_daic=10',
             'kernel_size_1_daic = [30,5]', 'kernel_size_2_daic = [3,257]',
             'stretch_factors=[(0.95, 1.),(1.053,1.)]',
             'output_type="pooled_map"', 'stretch_penality_lambda=0.', 'fc_insize=16779600',
             'training_mode="train_and_eval"', 'network_type= "2_layer"']

experiments[5] = ['layer_type="multi"', 'channels1_daic=10', 'channels2_daic=10',
             'kernel_size_1_daic = [30,5]', 'kernel_size_2_daic = [3,257]',
             'stretch_factors=[(0.9, 1.),(1.111,1.)]',
             'output_type="pooled_map"', 'stretch_penality_lambda=0.', 'fc_insize=16779600',
             'training_mode="train_and_eval"', 'network_type= "2_layer"']

experiments[6] = ['layer_type="multi"', 'channels1_daic=10', 'channels2_daic=10',
             'kernel_size_1_daic = [30,5]', 'kernel_size_2_daic = [3,257]',
             'stretch_factors=[(0.8, 1.),(1.25,1.)]',
             'output_type="pooled_map"', 'stretch_penality_lambda=0.', 'fc_insize=16779600',
             'training_mode="train_and_eval"', 'network_type= "2_layer"']

experiments[7] = ['layer_type="multi"', 'channels1_daic=10', 'channels2_daic=10',
             'kernel_size_1_daic = [30,5]', 'kernel_size_2_daic = [3,257]',
             'stretch_factors=[(0.7, 1.),(1.428,1.)]',
             'output_type="pooled_map"', 'stretch_penality_lambda=0.', 'fc_insize=16779600',
             'training_mode="train_and_eval"', 'network_type= "2_layer"']

experiments[8] = ['layer_type="multi"', 'channels1_daic=10', 'channels2_daic=10',
             'kernel_size_1_daic = [30,5]', 'kernel_size_2_daic = [3,257]',
             'stretch_factors=[(0.5, 1.),(2.,1.)]',
             'output_type="pooled_map"', 'stretch_penality_lambda=0.', 'fc_insize=16779600',
             'training_mode="train_and_eval"', 'network_type= "2_layer"']

experiments[9] = ['layer_type="multi"', 'channels1_daic=10', 'channels2_daic=10',
             'kernel_size_1_daic = [30,5]', 'kernel_size_2_daic = [3,257]',
             'stretch_factors=[(0.8, 1.),(0.9, 1.),(1.111, 1.),(1.25, 1.)]',
             'output_type="pooled_map"', 'stretch_penality_lambda=0.', 'fc_insize=16779600',
             'training_mode="train_and_eval"', 'network_type= "2_layer"']

experiments[10] = ['layer_type="multi"', 'channels1_daic=10', 'channels2_daic=10',
             'kernel_size_1_daic = [30,5]', 'kernel_size_2_daic = [3,257]',
             'stretch_factors=[(0.25, 1.),(0.5, 1.),(2.,1.),(4.,1.)]',
             'output_type="pooled_map"', 'stretch_penality_lambda=0.', 'fc_insize=16779600',
             'training_mode="train_and_eval"', 'network_type= "2_layer"']

experiments[11] = ['layer_type="multi"', 'channels1_daic=10', 'channels2_daic=10',
             'kernel_size_1_daic = [30,5]', 'kernel_size_2_daic = [3,257]',
             'stretch_factors=[(0.7, 1.),(0.8, 1.),(0.9, 1.),(1.111, 1.),(1.25, 1.),(1.428, 1.)]',
             'output_type="pooled_map"', 'stretch_penality_lambda=0.', 'fc_insize=16779600',
             'training_mode="train_and_eval"', 'network_type= "2_layer"']

experiments[12] = ['layer_type="multi"', 'channels1_daic=10', 'channels2_daic=10',
             'kernel_size_1_daic = [30,5]', 'kernel_size_2_daic = [3,257]',
             'stretch_factors=[(0.25, 1.),(0.5, 1.),(0.7, 1.),(1.428, 1.),(2., 1.),(4., 1.)]',
             'output_type="pooled_map"', 'stretch_penality_lambda=0.', 'fc_insize=16779600',
             'training_mode="train_and_eval"', 'network_type= "2_layer"']

#run experiments
short_description = 'initial test'
dataset = 'daic'
num_experiment = 1
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
    end = 12
    gpu_ID = 1


for num_run in range(begin,end+1):
    temp_params = '/'.join(experiments[num_run])
    xval.run_experiment(num_experiment,num_run,num_folds,dataset,experiment_folder,temp_params, gpu_ID)
