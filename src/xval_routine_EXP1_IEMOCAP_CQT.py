from __future__ import print_function
import numpy as np
import sys, os
import xval_instance as xval

#EXPERIMENT PARAMETERS:
gpu_ID = 0
overwrite_results = False  #if true overwrite existing experiment instances
debug_mode = False  #if false, if an error occurs in one instance, it is skipped without stopping the routine
short_description = 'testing learning rate on OMG_model, ParallelConv and AlexNet. No Augmentation'
dataset = 'iemocap_cqt'
task_type = 'classification'
generator = True
num_experiment = 1  #id of the experiment
num_folds = 1  #number of k-folds for cross-validation
experiment_folder = '../../../copy/consultancy/iemocap_cqt'  #where to save results

#DEFINE HERE EVERY INSTANCE OF THE EXPERIMENT
#every instance must be a key in the experiment dict
#every key must be a list of strings
#every parameter overwrites the correspective default parameter
#mandatory parameters:
#-task_type: classification, or regression
#-reshaping type: conv, lstm, none
#-architecture: one of the models defined in the models_API script
#-comment_1 and comment_2: write here any info you want to show in the results spreadsheet. Example: L2 increased to 0.1
experiment = {}
experiment[1] = ['architecture="OMG_model"', 'reshaping_type="cnn"',
                 'comment_1="lr 0.000001"', 'comment_2="OMG_model"',
                 'learning_rate=0.000001', 'output_classes=4']
experiment[2] = ['architecture="OMG_model"', 'reshaping_type="cnn"',
                 'comment_1="lr 0.00001"', 'comment_2="OMG_model"',
                 'learning_rate=0.00001', 'output_classes=4']
experiment[3] = ['architecture="OMG_model"', 'reshaping_type="cnn"',
                 'comment_1="lr 0.0001"', 'comment_2="OMG_model"',
                 'learning_rate=0.0001', 'output_classes=4']
experiment[4] = ['architecture="OMG_model"', 'reshaping_type="cnn"',
                 'comment_1="lr 0.0003"', 'comment_2="OMG_model"',
                 'learning_rate=0.0003', 'output_classes=4']
experiment[5] = ['architecture="OMG_model"', 'reshaping_type="cnn"',
                 'comment_1="lr 0.0006"', 'comment_2="OMG_model"',
                 'learning_rate=0.0006', 'output_classes=4']
experiment[6] = ['architecture="OMG_model"', 'reshaping_type="cnn"',
                 'comment_1="lr 0.001"', 'comment_2="OMG_model"',
                 'learning_rate=0.001', 'output_classes=4']
experiment[7] = ['architecture="OMG_model"', 'reshaping_type="cnn"',
                 'comment_1="lr 0.003"', 'comment_2="OMG_model"',
                 'learning_rate=0.003', 'output_classes=4']
experiment[8] = ['architecture="OMG_model"', 'reshaping_type="cnn"',
                 'comment_1="lr 0.005"', 'comment_2="OMG_model"',
                 'learning_rate=0.005', 'output_classes=4']

experiment[9] = ['architecture="AlexNet"', 'reshaping_type="cnn"',
                 'comment_1="lr 0.000001"', 'comment_2="AlexNet"',
                 'learning_rate=0.000001', 'output_classes=4']
experiment[10] = ['architecture="AlexNet"', 'reshaping_type="cnn"',
                 'comment_1="lr 0.00001"', 'comment_2="AlexNet"',
                 'learning_rate=0.00001', 'output_classes=4']
experiment[11] = ['architecture="AlexNet"', 'reshaping_type="cnn"',
                 'comment_1="lr 0.0001"', 'comment_2="AlexNet"',
                 'learning_rate=0.0001', 'output_classes=4']
experiment[12] = ['architecture="AlexNet"', 'reshaping_type="cnn"',
                 'comment_1="lr 0.0003"', 'comment_2="AlexNet"',
                 'learning_rate=0.0003', 'output_classes=4']
experiment[13] = ['architecture="AlexNet"', 'reshaping_type="cnn"',
                 'comment_1="lr 0.0006"', 'comment_2="AlexNet"',
                 'learning_rate=0.0006', 'output_classes=4']
experiment[14] = ['architecture="AlexNet"', 'reshaping_type="cnn"',
                 'comment_1="lr 0.001"', 'comment_2="AlexNet"',
                 'learning_rate=0.001', 'output_classes=4']
experiment[15] = ['architecture="AlexNet"', 'reshaping_type="cnn"',
                 'comment_1="lr 0.003"', 'comment_2="AlexNet"',
                 'learning_rate=0.003', 'output_classes=4']
experiment[16] = ['architecture="AlexNet"', 'reshaping_type="cnn"',
                 'comment_1="lr 0.005"', 'comment_2="AlexNet"',
                 'learning_rate=0.005', 'output_classes=4']

experiment[17] = ['architecture="ParallelConv"', 'reshaping_type="cnn"',
                 'comment_1="lr 0.000001"', 'comment_2="ParallelConv"',
                 'learning_rate=0.000001', 'output_classes=4']
experiment[18] = ['architecture="ParallelConv"', 'reshaping_type="cnn"',
                 'comment_1="lr 0.00001"', 'comment_2="ParallelConv"',
                 'learning_rate=0.00001', 'output_classes=4']
experiment[19] = ['architecture="ParallelConv"', 'reshaping_type="cnn"',
                 'comment_1="lr 0.0001"', 'comment_2="ParallelConv"',
                 'learning_rate=0.0001', 'output_classes=4']
experiment[20] = ['architecture="ParallelConv"', 'reshaping_type="cnn"',
                 'comment_1="lr 0.0003"', 'comment_2="ParallelConv"',
                 'learning_rate=0.0003', 'output_classes=4']
experiment[21] = ['architecture="ParallelConv"', 'reshaping_type="cnn"',
                 'comment_1="lr 0.0006"', 'comment_2="ParallelConv"',
                 'learning_rate=0.0006', 'output_classes=4']
experiment[22] = ['architecture="ParallelConv"', 'reshaping_type="cnn"',
                 'comment_1="lr 0.001"', 'comment_2="ParallelConv"',
                 'learning_rate=0.001', 'output_classes=4']
experiment[23] = ['architecture="ParallelConv"', 'reshaping_type="cnn"',
                 'comment_1="lr 0.003"', 'comment_2="ParallelConv"',
                 'learning_rate=0.003', 'output_classes=4']
experiment[24] = ['architecture="ParallelConv"', 'reshaping_type="cnn"',
                 'comment_1="lr 0.005"', 'comment_2="ParallelConv"',
                 'learning_rate=0.005', 'output_classes=4']

#DON'T TOUCH WHAT IS WRITTEN BELOW THIS LINE
#-------------------------------------------------------------------------------#
print ('Overwrite results: ' + str(overwrite_results))
print ('Debug mode: ' + str(debug_mode))

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
    #if not specified run all experiments
    keys = list(experiment.keys())
    begin = keys[0]
    end = keys[-1]

for num_run in range(begin,end+1):
    results_name = output_path + '/results/results_' + dataset + '_exp' + str(num_experiment) + '_run' + str(num_run) + '.npy'
    temp_params = '/'.join(experiment[num_run])

    if overwrite_results:
        if debug_mode == False:
            try:
                xval.run_experiment(num_experiment,num_run,num_folds,dataset,experiment_folder,temp_params, gpu_ID, task_type, generator)
            except:
                pass
        else:
            xval.run_experiment(num_experiment,num_run,num_folds,dataset,experiment_folder,temp_params, gpu_ID, task_type, generator)

    else:  #if not overwrite results
        if not os.path.exists(output_path):  #not overwrite experiments
            if debug_mode == False:
                try:
                    xval.run_experiment(num_experiment,num_run,num_folds,dataset,experiment_folder,temp_params, gpu_ID, task_type, generator)
                except:
                    pass
            else:
                xval.run_experiment(num_experiment,num_run,num_folds,dataset,experiment_folder,temp_params, gpu_ID, task_type, generator)
        else:  #if result exists print below line
            print ('exp' + str(num_experiment) + ' run' + str(num_run) + ' already exists: skipping')
print('')
print ('REQUESTED EXPERIMENTS SUCCESSFULLY COMPLETED')
