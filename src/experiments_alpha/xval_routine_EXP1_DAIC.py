from __future__ import print_function
import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import xval_instance as xval

#EXPERIMENT PARAMETERS:
short_description = 'stupid test'
dataset = 'daic'
num_experiment = 1  #id of the experiment
num_folds = 4  #number of k-folds for cross-validation
#experiment_folder = '../../../copy/prova_API'  #where to save results
experiment_folder = '../results_daic_prova'  #where to save results

#DEFINE HERE EVERY INSTANCE OF THE EXPERIMENT
#every instance must be a key in the experiment dict
#every key must be a list of stringsprova_API
#every parameter overwrites the default parameters
#mandatory parameters:
#-task_type: multimodal_classification, binary_classification or regression
#-architecture: one of the models defined in the models_API script
#-comment: write here any info you want to show in the results spreadsheet. Example: L2 increased to 0.1
experiment = {}
experiment[1] = ['task_type= "regression"', 'architecture="EXAMPLE_model_regression"',
                 'generator = False', 'comment_1="init "', 'comment_2="EXAMPLE_architecture"']


#DON'T TOUCH WHAT IS WRITTEN BELOW THIS LINE
#-------------------------------------------------------------------------------
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
    temp_params = '/'.join(experiment[num_run])
    xval.run_experiment(num_experiment,num_run,num_folds,dataset,experiment_folder,temp_params, gpu_ID)

print ('REQUESTED EXPERIMENTS SUCCESSFULLY COMPLETED')
