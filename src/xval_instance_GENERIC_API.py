import numpy as np
import os, sys
import subprocess
import time
import shutil


def save_code(output_code_path):
    curr_src_path = './'
    curr_config_path = '../config/'
    output_src_path = output_code_path + '/src'
    output_config_path = output_code_path + '/config'
    line1 = 'cp ' + curr_src_path + '* ' + output_src_path
    line2 = 'cp ' + curr_config_path + '* ' + output_config_path
    copy1 = subprocess.Popen(line1, shell=True)
    copy1.communicate()
    copy2 = subprocess.Popen(line2, shell=True)
    copy2.communicate()


def run_experiment(num_experiment, num_run, num_folds, dataset, experiment_folder, parameters, gpu_ID):
    '''
    run the crossvalidation
    '''
    print("NEW EXPERIMENT: exp: " + str(num_experiment) + ' run: ' + str(num_run))
    print('Dataset: ' + dataset)

    #create output path if not existing
    output_path = experiment_folder + '/experiment_' + str(num_experiment)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_temp_path = output_path + '/temp'
    if not os.path.exists(output_temp_path):
        os.makedirs(output_temp_path)

    output_models_path = output_path + '/models'
    if not os.path.exists(output_models_path):
        os.makedirs(output_models_path)

    output_results_path = output_path + '/results'
    if not os.path.exists(output_results_path):
        os.makedirs(output_results_path)

    output_temp_data_path = output_temp_path + '/temp_data'
    if not os.path.exists(output_temp_data_path):
        os.makedirs(output_temp_data_path)

    output_temp_results_path = output_temp_path + '/temp_results'
    if not os.path.exists(output_temp_results_path):
        os.makedirs(output_temp_results_path)

    output_code_path = output_path + '/code'
    if not os.path.exists(output_code_path):
        os.makedirs(output_code_path)

    output_src_path = output_code_path + '/src'
    if not os.path.exists(output_src_path):
        os.makedirs(output_src_path)

    output_config_path = output_code_path + '/config'
    if not os.path.exists(output_config_path):
        os.makedirs(output_config_path)


    #initialize results dict
    folds = {}

    #iterate folds
    for i in range(num_folds):
        #create paths
        num_fold = i

        #init paths
        model_name = output_models_path + '/model_xval_' + dataset + '_exp' + str(num_experiment) + '_run' + str(num_run) + '_fold' + str(num_fold)
        results_name = output_temp_results_path + '/temp_results_' + dataset + '_exp' + str(num_experiment) + '_run' + str(num_run) + '_fold' + str(num_fold) + '.npy'

        #init results as ERROR
        np.save(results_name, np.array(['ERROR']))

        #run training
        training = subprocess.Popen(['python3', 'build_model_GENERIC_keras.py',
                                     'crossvalidation', str(num_experiment), str(num_run),
                                      str(num_fold), parameters, model_name, results_name,
                                      output_temp_data_path, dataset, str(gpu_ID), str(num_folds)])
        training.communicate()
        training.wait()

        #wait for file to be created
        flag = 'ERROR'
        while flag == 'ERROR':
            time.sleep(0.2)
            flag = np.load(results_name)

        #update results dict
        temp_results = np.load(results_name)
        temp_results = temp_results.item()
        folds[i] = temp_results

    #compute summary
    folds['summary'] = {}
    folds['summary']['parameters'] = parameters

    #compute mean loss and loss std
    tr_loss = []
    val_loss = []
    test_loss = []
    for i in range(num_folds):
        tr_loss.append(folds[i]['train_loss'])
        val_loss.append(folds[i]['val_loss'])
        test_loss.append(folds[i]['test_loss'])
    tr_mean = np.mean(tr_loss)
    val_mean = np.mean(val_loss)
    test_mean = np.mean(test_loss)
    tr_std = np.std(tr_loss)
    val_std = np.std(val_loss)
    test_std = np.std(test_loss)
    folds['summary'] = {'training':{'mean_loss': tr_mean,
                                    'loss_std': tr_std},
                        'validation':{'mean_loss': val_mean,
                                    'loss_std': val_std},
                        'test':{'mean_loss': test_mean,
                                    'loss_std': test_std}}

    #compute mean acc and acc std if task_type is classification

    #unroll parameters to find task_type:
    unrolled = parameters.split('/')
    for param in unrolled:
        print (param)
        if 'task_type' in param:
            print (param)
            exec(param)
    if task_type != 'regression':
        tr_acc = []
        val_acc = []
        test_acc = []
        for i in range(num_folds):
            tr_acc.append(folds[i]['train_acc'])
            val_acc.append(folds[i]['val_acc'])
            test_acc.append(folds[i]['test_acc'])
        tr_mean = np.mean(tr_acc)
        val_mean = np.mean(val_acc)
        test_mean = np.mean(test_acc)
        tr_std = np.std(tr_acc)
        val_std = np.std(val_acc)
        test_std = np.std(test_acc)
        folds['summary'] = {'training':{'mean_acc': tr_mean,
                                        'acc_std': tr_std},
                            'validation':{'mean_acc': val_mean,
                                        'acc_std': val_std},
                            'test':{'mean_acc': test_mean,
                                        'loss_std': test_std}}


    print (folds)

    #save results dict
    dict_name = 'results_' + dataset + '_exp' + str(num_experiment) + '_run' + str(num_run) + '.npy'
    final_dict_path = output_results_path + '/' + dict_name
    np.save(final_dict_path, folds)

    #save current code
    save_code(output_code_path)


if __name__ == '__main__':
    run_experiment()
