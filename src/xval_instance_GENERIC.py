import numpy as np
import os, sys
import subprocess
import preprocessing_DAIC as pre
import time
import shutil


def folds_generator_daic(num_folds):

    sequence = pre.get_sequence()
    dup_sequence = sequence * (num_folds+1)
    shift = len(sequence) / num_folds
    fold_list = {}
    for i in range(num_folds):
        curr_shift = int(shift * i)
        #print (curr_shift)
        curr_sequence = dup_sequence[curr_shift:curr_shift+len(sequence)]
        tr, val, test = pre.gen_split_lists(curr_sequence)

        fold_list[i] = {'train': tr,
                          'val': val,
                          'test': test}

    return fold_list

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


def init_experiment_dict(num_folds):
    '''init dict for crossvalidation experiments'''
    folds = {}
    for i in np.arange(num_folds):
        #temp_exp_name = 'experiment_' + str(i+1)
        folds[i] = {'training':{},
                                    'validation':{},
                                    'test':{}}

    return folds

def build_experiment_dict(n_folds, n_actors, dataset):
    '''fill dict with values for a desired experiment'''
    #ac_list = folds_generator(n_folds, n_actors, dataset)
    if dataset == 'daic':
        ac_list = folds_generator_daic(n_folds)
    folds = init_experiment_dict(n_folds)

    for i in range(n_folds):
        folds[i]['training']['actors'] = ac_list[i]['train']
        folds[i]['validation']['actors'] = ac_list[i]['val']
        folds[i]['test']['actors'] = ac_list[i]['test']

    return folds, ac_list

def run_experiment(num_experiment, num_run, num_folds, dataset, experiment_folder, parameters, gpu_ID):
    '''
    run the crossvalidation
    '''
    print("NEW EXPERIMENT: exp: " + str(num_experiment) + ' run: ' + str(num_run))
    print('Dataset: ' + dataset)

    if dataset == 'daic':
        num_actors = 142

    else:
        raise ValueError('Invalid dataset name')


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


    #create dict wit actors distributed per every fold
    folds, folds_list = build_experiment_dict(num_folds, num_actors, dataset)


    #iterate folds
    for i in range(num_folds):
        #create paths
        num_fold = i

        #init paths to give to build model script
        model_name = output_models_path + '/model_xval_' + dataset + '_exp' + str(num_experiment) + '_run' + str(num_run) + '_fold' + str(num_fold)
        results_name = output_temp_results_path + '/temp_results_' + dataset + '_exp' + str(num_experiment) + '_run' + str(num_run) + '_fold' + str(num_fold) + '.npy'

        #init results as ERROR
        np.save(results_name, np.array(['ERROR']))

        folds_list = str(folds_list)  #convert to string to pass it as argument

        #run training
        training = subprocess.Popen(['python3', 'build_model_GENERIC_multiscale_auto.py',
                                     'crossvalidation', str(num_experiment), str(num_run),
                                      str(num_fold), parameters, model_name, results_name,
                                      output_temp_data_path, dataset, str(gpu_ID), folds_list])
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
    tr_loss = []
    val_loss = []
    test_loss = []
    for i in range(num_folds):
        tr_loss.append(folds[i]['train_loss_BVL'])
        val_loss.append(folds[i]['val_loss_BVL'])
        test_loss.append(folds[i]['test_loss_BVL'])
    tr_mean = np.mean(tr_loss)
    val_mean = np.mean(val_loss)
    test_mean = np.mean(test_loss)
    tr_dev = np.std(tr_loss)
    val_dev = np.std(val_loss)
    test_dev = np.std(test_loss)
    folds['summary'] = {'training_BVL':{'mean_loss': tr_mean,
                                    'loss_std': tr_dev},
                        'validation_BVL':{'mean_loss': val_mean,
                                    'loss_std': val_dev},
                        'test_BVL':{'mean_loss': test_mean,
                                    'loss_std': test_dev}}
    folds['summary']['parameters'] = parameters

    print (folds)

    #save results dict
    dict_name = 'results_' + dataset + '_exp' + str(num_experiment) + '_run' + str(num_run) + '.npy'
    final_dict_path = output_results_path + '/' + dict_name
    np.save(final_dict_path, folds)

    #save current code
    save_code(output_code_path)

    #remove big temp files
    os.remove(TRAINING_PREDICTORS_P)
    os.remove(TRAINING_TARGET_P)
    os.remove(VALIDATION_PREDICTORS_P)
    os.remove(VALIDATION_TARGET_P)
    os.remove(TEST_PREDICTORS_P)
    os.remove(TEST_TARGET_P)

if __name__ == '__main__':
    run_experiment()
