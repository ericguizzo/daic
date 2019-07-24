import numpy as np
import os, sys
import subprocess
import time
import shutil

tr_perc = 0.7
val_perc = 0.2

def folds_generator(num_folds, num_actors, dataset):
    ac_list = list(range(num_actors)) * num_folds
    n_train = int(num_actors * tr_perc)
    n_val = int(num_actors * val_perc)
    n_test = int(num_actors - (n_train + n_val))
    if dataset == 'digits':
        n_train = 2
        n_val = 1
        n_test = 1
    if dataset == 'cifar':
        n_train = 7
        n_val = 2
        n_test = 1
    if dataset == 'mnist':
        n_train = 7
        n_val = 2
        n_test = 1
    shift = num_actors / num_folds
    fold_actors_list = {}
    for i in range(num_folds):
        curr_shift = int(shift * i)
        tr_ac = ac_list[curr_shift:curr_shift+n_train]
        val_ac = ac_list[curr_shift+n_train:curr_shift+n_train+n_val]
        test_ac = ac_list[curr_shift+n_train+n_val:curr_shift+n_train+n_val+n_test]
        fold_actors_list[i] = {'train': tr_ac,
                          'val': val_ac,
                          'test': test_ac}

    return fold_actors_list

def get_fold_dataset(actors_list, merged_predictors, merged_target, dataset):
    if dataset == 'speechCmd':
        predictors = np.array([])
        target = np.array([])
        for i in actors_list:
            print (i, predictors.shape)
            if i == actors_list[0]:
                predictors = merged_predictors[i]
                target = merged_target[i]
                print (i, predictors.shape)

            else:
                predictors = np.concatenate((predictors, merged_predictors[i]), axis=0)
                target = np.concatenate((target, merged_target[i]), axis=0)
    elif dataset == 'ravdess':
        predictors = []
        target = []
        for i in actors_list:
            predictors.append(merged_predictors[i])
            target.append(merged_target[i])
        predictors = np.array(predictors)
        target = np.array(target)
        predictors = predictors.reshape(predictors.shape[0]*predictors.shape[1],
                                        predictors.shape[2], predictors.shape[3])
        target = target.reshape(target.shape[0]*target.shape[1], target.shape[2])

    elif dataset == 'digits':
        predictors = np.array([])
        target = np.array([])
        for i in actors_list:
            print (i, predictors.shape)
            if i == actors_list[0]:
                predictors = merged_predictors[i]
                target = merged_target[i]
                print (i, predictors.shape)

            else:
                predictors = np.concatenate((predictors, merged_predictors[i]), axis=0)
                target = np.concatenate((target, merged_target[i]), axis=0)

    elif dataset == 'cifar':
        predictors = np.array([])
        target = np.array([])
        for i in actors_list:
            print (i, predictors.shape)
            if i == actors_list[0]:
                predictors = merged_predictors[i]
                target = merged_target[i]
                print (i, predictors.shape)

            else:
                predictors = np.concatenate((predictors, merged_predictors[i]), axis=0)
                target = np.concatenate((target, merged_target[i]), axis=0)

    elif dataset == 'mnist':
        predictors = np.array([])
        target = np.array([])
        for i in actors_list:
            print (i, predictors.shape)
            if i == actors_list[0]:
                predictors = merged_predictors[i]
                target = merged_target[i]
                print (i, predictors.shape)

            else:
                predictors = np.concatenate((predictors, merged_predictors[i]), axis=0)
                target = np.concatenate((target, merged_target[i]), axis=0)


    print ('predictors shape:' + str(predictors.shape))
    print ('target shape:' + str(target.shape))

    return predictors, target

def save_fold_datasets(tr_list, val_list, test_list, merged_predictors, merged_target, dataset, num_experiment, num_run, num_fold):
    tr_pred, tr_tg = get_fold_dataset(tr_list, merged_predictors, merged_target, dataset)
    val_pred, val_tg = get_fold_dataset(val_list, merged_predictors, merged_target, dataset)
    test_pred, test_tg = get_fold_dataset(test_list, merged_predictors, merged_target, dataset)
    np.save(TRAINING_PREDICTORS_P, tr_pred)
    np.save(TRAINING_TARGET_P, tr_tg)
    np.save(VALIDATION_PREDICTORS_P, val_pred)
    np.save(VALIDATION_TARGET_P, val_tg)
    np.save(TEST_PREDICTORS_P, test_pred)
    np.save(TEST_TARGET_P, test_tg)


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
    ac_list = folds_generator(n_folds, n_actors, dataset)
    folds = init_experiment_dict(n_folds)

    for i in range(n_folds):
        folds[i]['training']['actors'] = ac_list[i]['train']
        folds[i]['validation']['actors'] = ac_list[i]['val']
        folds[i]['test']['actors'] = ac_list[i]['test']

    return folds

def run_experiment(num_experiment, num_run, num_folds, dataset, experiment_folder, parameters, gpu_ID):
    '''
    run the crossvalidation
    '''
    print("NEW EXPERIMENT: exp: " + str(num_experiment) + ' run: ' + str(num_run))
    print('Dataset: ' + dataset)

    if dataset == 'ravdess':
        num_actors = 24
    elif dataset == 'speechCmd':
        num_actors = 400
    elif dataset == 'tess':
        num_actors = 2
    elif dataset == 'digits':
        num_actors = 2
    elif dataset == 'cifar':
        num_actors = 10
    elif dataset == 'mnist':
        num_actors = 10
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

    #load dataset
    if dataset == 'ravdess':
        merged_predictors = np.load('../dataset/matrices/merged_crossval_predictors.npy')
        merged_target = np.load('../dataset/matrices/merged_crossval_target.npy')
        merged_predictors = merged_predictors.item()
        merged_target = merged_target.item()
    elif dataset == 'digits':
        merged_predictors = np.load('../dataset/matrices/merged_digits_predictors.npy')
        merged_target = np.load('../dataset/matrices/merged_digits_target.npy')
        merged_predictors = merged_predictors.item()
        merged_target = merged_target.item()
    elif dataset == 'cifar':
        merged_predictors = np.load('../dataset/matrices/merged_cifar_predictors.npy')
        merged_target = np.load('../dataset/matrices/merged_cifar_target.npy')
        merged_predictors = merged_predictors.item()
        merged_target = merged_target.item()
    elif dataset == 'speechCmd':
        merged_predictors = np.load('../dataset/matrices/merged_speechCmd_predictors.npy')
        merged_target = np.load('../dataset/matrices/merged_speechCmd_target.npy')
        merged_predictors = merged_predictors.item()
        merged_target = merged_target.item()
    elif dataset == 'mnist':
        merged_predictors = np.load('../dataset/matrices/merged_mnist_predictors.npy')
        merged_target = np.load('../dataset/matrices/merged_mnist_target.npy')
        merged_predictors = merged_predictors.item()
        merged_target = merged_target.item()
    elif dataset == 'tess':
        pass
    else:
        raise ValueError('Invalid dataset selected')


    #create dict wit actors distributed per every fold
    folds = build_experiment_dict(num_folds, num_actors, dataset)

    #iterate folds
    for i in range(num_folds):
        #create paths
        num_fold = i
        # set temp dataset save path
        TRAINING_PREDICTORS_P = output_temp_data_path + '/xval_' + dataset + '_predictors_tr_' + str(num_experiment) + str(num_run) + str(num_fold) + '.npy'
        TRAINING_TARGET_P = output_temp_data_path + '/xval_' + dataset + '_target_tr_' + str(num_experiment) + str(num_run) + str(num_fold) + '.npy'
        VALIDATION_PREDICTORS_P = output_temp_data_path + '/xval_' + dataset + '_predictors_v_' + str(num_experiment) + str(num_run) + str(num_fold) + '.npy'
        VALIDATION_TARGET_P = output_temp_data_path + '/xval_' + dataset + '_target_v_' + str(num_experiment) + str(num_run) + str(num_fold) + '.npy'
        TEST_PREDICTORS_P = output_temp_data_path + '/xval_' + dataset + '_predictors_ts_' + str(num_experiment) + str(num_run) + str(num_fold) + '.npy'
        TEST_TARGET_P = output_temp_data_path + '/xval_' + dataset + '_target_ts_' + str(num_experiment) + str(num_run) + str(num_fold) + '.npy'

        #init paths to give to build model script
        model_name = output_models_path + '/model_xval_' + dataset + '_exp' + str(num_experiment) + '_run' + str(num_run) + '_fold' + str(num_fold)
        results_name = output_temp_results_path + '/temp_results_' + dataset + '_exp' + str(num_experiment) + '_run' + str(num_run) + '_fold' + str(num_fold) + '.npy'
        #delete existing datasets (to be sure that the training gets the newest one)
        np.save(TRAINING_PREDICTORS_P, np.array(['ERROR']))
        np.save(TRAINING_TARGET_P, np.array(['ERROR']))
        np.save(VALIDATION_PREDICTORS_P, np.array(['ERROR']))
        np.save(VALIDATION_TARGET_P, np.array(['ERROR']))
        np.save(TEST_PREDICTORS_P, np.array(['ERROR']))
        np.save(TEST_TARGET_P, np.array(['ERROR']))
        np.save(results_name, np.array(['ERROR']))


        #compute and save dataset
        if dataset == 'tess':
            pass
        else:
            tr_actors = folds[i]['training']['actors']
            val_actors = folds[i]['validation']['actors']
            test_actors = folds[i]['test']['actors']
            tr_pred, tr_tg = get_fold_dataset(tr_actors, merged_predictors, merged_target, dataset)
            val_pred, val_tg = get_fold_dataset(val_actors, merged_predictors, merged_target, dataset)
            test_pred, test_tg = get_fold_dataset(test_actors, merged_predictors, merged_target, dataset)
            np.save(TRAINING_PREDICTORS_P, tr_pred)
            np.save(TRAINING_TARGET_P, tr_tg)
            np.save(VALIDATION_PREDICTORS_P, val_pred)
            np.save(VALIDATION_TARGET_P, val_tg)
            np.save(TEST_PREDICTORS_P, test_pred)
            np.save(TEST_TARGET_P, test_tg)

        #save_fold_datasets(tr_actors, val_actors, test_actors, merged_predictors, merged_target, dataset,
        #                    num_experiment, num_run, num_fold)

        #run training
        training = subprocess.Popen(['python3', 'build_model_GENERIC_multiscale_auto.py',
                                     'crossvalidation', str(num_experiment), str(num_run),
                                      str(num_fold), parameters, model_name, results_name,
                                      output_temp_data_path, dataset, str(gpu_ID)])
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
    tr_acc = []
    val_acc = []
    test_acc = []
    for i in range(num_folds):
        tr_acc.append(folds[i]['train_acc_BVL'])
        val_acc.append(folds[i]['val_acc_BVL'])
        test_acc.append(folds[i]['test_acc_BVL'])
    tr_mean = np.mean(tr_acc)
    val_mean = np.mean(val_acc)
    test_mean = np.mean(test_acc)
    tr_dev = np.std(tr_acc)
    val_dev = np.std(val_acc)
    test_dev = np.std(test_acc)
    folds['summary'] = {'training_BVL':{'mean_accuracy': tr_mean,
                                    'acc_std': tr_dev},
                        'validation_BVL':{'mean_accuracy': val_mean,
                                    'acc_std': val_dev},
                        'test_BVL':{'mean_accuracy': test_mean,
                                    'acc_cd std': test_dev}}
    folds['summary']['parameters'] = parameters

    tr_acc = []
    val_acc = []
    test_acc = []
    for i in range(num_folds):
        tr_acc.append(folds[i]['train_acc_BVA'])
        val_acc.append(folds[i]['val_acc_BVA'])
        test_acc.append(folds[i]['test_acc_BVA'])
    tr_mean = np.mean(tr_acc)
    val_mean = np.mean(val_acc)
    test_mean = np.mean(test_acc)
    tr_dev = np.std(tr_acc)
    val_dev = np.std(val_acc)
    test_dev = np.std(test_acc)
    folds['summary']['training_BVA'] = {'mean_accuracy': tr_mean,
                                        'acc_std': tr_dev}
    folds['summary']['validation_BVA'] = {'mean_accuracy': val_mean,
                                          'acc_Std': val_dev}
    folds['summary']['test_BVA'] = {'mean_accuracy': test_mean,
                                    'acc_std': test_dev}

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
