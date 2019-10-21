import os, sys
import subprocess


#DEFINE PARAMETERS
num_experiments_to_run = []

#global_parameters = ['num_epochs=500', 'learning_rate=0.00005', 'early_stopping=True',
#                    'patience=10', 'choose_optimizer="adam"', 'batch_size=10']

global_parameters = ['num_epochs=20', 'learning_rate=0.00005', 'early_stopping=False',
                    'patience=10', 'choose_optimizer="adam"', 'batch_size=10']

experiments_to_run_folder = 'experiments_multiconv_SER'
'''
xval_script_parameters = ['experiment_folder="../../results_final_SER_LR000005_alexnet"',
                          'debug_mode=False',
                          'overwrite_results=False',
                          'num_folds=3']
'''

xval_script_parameters = ['experiment_folder="../../proooova"',
                          'debug_mode=True',
                          'overwrite_results=True',
                          'num_folds=1']
begin = 1

end = 16

gpu_ID = 0

try:
    num_experiments_to_run = eval(sys.argv[1])
    gpu_ID = int(sys.argv[2])
except:
    pass

try:
    begin = eval(sys.argv[3])
except:
    pass

try:
    end = sys.argv[4]
except:
    pass

contents = os.listdir(experiments_to_run_folder)
contents = [os.path.join(experiments_to_run_folder, x) for x in contents]

#convert to string to pass them through subprocess
global_parameters = '%'.join(global_parameters)
xval_script_parameters = '%'.join(xval_script_parameters)


#filter only experiments to run
def filter_experiments(to_run=num_experiments_to_run, ex_list=contents):
    filtered_list = []
    for exp in ex_list:
        id = int(exp.split('/')[-1].split('_')[0][2:])
        if id in to_run:
            filtered_list.append(exp)
    return filtered_list

filtered_list = filter_experiments()

for exp_name in filtered_list:
    process = subprocess.Popen(['python3', exp_name, str(begin), str(end), str(gpu_ID),
                                global_parameters, xval_script_parameters])

    process.communicate()
    process.wait()

print ('ALL EXPERIMENTS REQUESTED TO EXP MANAGER COMPLETED')
