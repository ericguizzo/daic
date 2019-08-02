from __future__ import print_function
import numpy as np
import sys, os
import xval_instance as xval

#EXPERIMENT PARAMETERS:
overwrite_results = False  #if true overwrite existing experiment instances
debug_mode = True  #if false, if an error occurs in one instance, it is skipped without stopping the routine
short_description = 'stupid test'
dataset = 'ravdess'
num_experiment = 1  #id of the experiment
num_folds = 2  #number of k-folds for cross-validation
#experiment_folder = '../../../copy/prova_API'  #where to save results
experiment_folder = '../results'  #where to save results

#DEFINE HERE EVERY INSTANCE OF THE EXPERIMENT
#every instance must be a key in the experiment dict
#every key must be a list of strings
#every parameter overwrites the correspective default parameter
#mandatory parameters:
#-task_type: classification, or regression
#-architecture: one of the models defined in the models_API script
#-comment_1 and comment_2: write here any info you want to show in the results spreadsheet. Example: L2 increased to 0.1
experiment = {}
experiment[1] = ['task_type= "classification"', 'architecture="EXAMPLE_model"',
                 'comment_1="reg 0.001"', 'comment_2="EXAMPLE_architecture"','regularization_lambda="0.1"']
experiment[2] = ['task_type= "classification"', 'architecture="EXAMPLE_model"',
                 'comment_1="reg 0.01"', 'comment_2="EXAMPLE_architecture"','regularization_lambda="0.01"']
experiment[3] = ['task_type= "classification"', 'architecture="EXAMPLE_model"',
                 'comment_1="reg 0.1"', 'comment_2="EXAMPLE_architecture"','regularization_lambda="0.1"']
experiment[4] = ['task_type= "classification"', 'architecture="EXAMPLE_model"',
                 'comment_1="reg 0.3"', 'comment_2="EXAMPLE_architecture"','regularization_lambda="0.1"']


#DON'T TOUCH WHAT IS WRITTEN BELOW THIS LINE
#-------------------------------------------------------------------------------#
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
    gpu_ID = 1

for num_run in range(begin,end+1):
    if overwrite_results:
        results_name = 'unexisting_path'
    else:
        results_name = output_path + '/results/results_' + dataset + '_exp' + str(num_experiment) + '_run' + str(num_run) + '.npy'
    temp_params = '/'.join(experiment[num_run])

    if not os.path.exists(output_path):  #not overwrite experiments
        if debug_mode == False:
            try:
                xval.run_experiment(num_experiment,num_run,num_folds,dataset,experiment_folder,temp_params, gpu_ID)
            except:
                pass
        else:
            xval.run_experiment(num_experiment,num_run,num_folds,dataset,experiment_folder,temp_params, gpu_ID)
    else:
        print ('exp' + str(num_exp) + ' run' + str(num_run) + ' already exists: skipping')

print ('REQUESTED EXPERIMENTS SUCCESSFULLY COMPLETED')
