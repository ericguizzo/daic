import numpy as np

from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras.utils import np_utils
from keras.backend import int_shape
from keras.models import load_model
from keras import regularizers
from keras import optimizers
import utility_functions as uf
from keras import backend as K
import models_API as choose_model
import matplotlib.pyplot as plt
#import preprocessing_DAIC as pre
import sys, os
import loadconfig
import configparser

#np.random.seed(0)
#torch.manual_seed(0)
print('')
config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

#load parameters from config file
DATASET_FOLDER = cfg.get('preprocessing', 'output_folder')
SAVE_MODEL = cfg.get('model', 'save_model') #only if not in crossval mode
SAVE_RESULTS = cfg.get('model', 'save_results') #only if not in crossval mode

#look at sys argv: if in crossvalidation model i/o matrices and new model filename
#are given from crossvalidation script, otherwise are normally taken from config.ini
try:
    cross_tag =  sys.argv[1]
    if cross_tag == 'crossvalidation':
        num_experiment = sys.argv[2]
        num_run = sys.argv[3]
        num_fold = sys.argv[4]
        parameters = sys.argv[5]
        model_path = sys.argv[6]
        results_path = sys.argv[7]
        output_temp_data_path = sys.argv[8]
        dataset = sys.argv[9]
        gpu_ID = int(sys.argv[10])
        num_folds = int(sys.argv[11])
        SAVE_MODEL = model_path

        print('crossvalidation mode: I/O from crossvalidation script')
        print('')
        print ('dataset: ' + dataset)
        print ('')
        print ('saving results at: ' + results_path)
        print('saving model at: ' + SAVE_MODEL + '.hdf5')
        print ('')

except IndexError:
    #test parameters
    #IF IN TEST MODE:no xvalidation, results saved as exp0
    dataset = cfg.get('model', 'dataset_test')
    architecture = 'OMG_model'
    parameters = '0,0'
    results_path = SAVE_RESULTS
    num_fold = 0
    num_exp = 0
    num_run = 0
    num_folds = 1

    print ('test mode: I/O from config.ini file')
    print ('')
    print ('dataset: ' + dataset)
    print ('')
    print ('saving results at: ' + SAVE_RESULTS)
    print ('')
    print ('saving model at: ' + SAVE_MODEL + '.hdf5')
    print ('')


predictors_name = dataset + '_predictors.npy'
target_name = dataset + '_target.npy'
PREDICTORS_LOAD = os.path.join(DATASET_FOLDER, predictors_name)
TARGET_LOAD = os.path.join(DATASET_FOLDER, target_name)

#default training parameters
train_split = 0.7
validation_split = 0.2
test_split = 0.1
percs = [train_split, validation_split, test_split]
save_best_model_metric = 'val_loss'
save_best_model_mode = 'min'
gpu_ID = 1
early_stopping = True
patience = 10
batch_size = 120
num_epochs = 2
learning_rate = 0.0005
regularization_lambda = 0.07
conv_regularization_lambda = 0.01
task_type = 'multilabel_classification'

if task_type == 'multilabel_classification':
    loss_function = 'categorical_crossentropy'
    metrics_list = ['accuracy']
elif task_type == 'binary_classification':
    loss_function = 'binary_crossentropy'
    metrics_list = ['accuracy']
elif task_type == 'regression':
    loss_function = 'MSE'
    metrics_list = []

else:
    raise ValueError('task_type can be only: multilabel_classification, binary_classification or regression')


#path for saving best val loss and best val acc models
BVL_model_path = SAVE_MODEL + '.hdf5'

#OVERWRITE DEFAULT PARAMETERS IF IN XVAL MODE
try:
    a = sys.argv[5]
    parameters = parameters.split('/')
    for param in parameters:
        exec(param)

except IndexError:
    pass

#define optimizer
opt = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

def main():

    #CREATE DATASET
    #load numpy data
    print('\n loading dataset...')

    folds_dataset_path = '../dataset/matrices'
    curr_fold_string = dataset + '_test_target_fold_' + str(num_fold) + '.npy'
    curr_fold_path = os.path.join(folds_dataset_path, curr_fold_string)

    train_pred_path = dataset + '_training_predictors_fold_' + str(num_fold) + '.npy'
    train_target_path = dataset + '_training_target_fold_' + str(num_fold) + '.npy'
    train_pred_path = os.path.join(folds_dataset_path, train_pred_path)
    train_target_path = os.path.join(folds_dataset_path, train_target_path)

    val_pred_path = dataset + '_validation_predictors_fold_' + str(num_fold) + '.npy'
    val_target_path = dataset + '_validation_target_fold_' + str(num_fold) + '.npy'
    val_pred_path = os.path.join(folds_dataset_path, val_pred_path)
    val_target_path = os.path.join(folds_dataset_path, val_target_path)

    test_pred_path = dataset + '_test_predictors_fold_' + str(num_fold) + '.npy'
    test_target_path = dataset + '_test_target_fold_' + str(num_fold) + '.npy'
    test_pred_path = os.path.join(folds_dataset_path, test_pred_path)
    test_target_path = os.path.join(folds_dataset_path, test_target_path)

    #compute which actors put in train, val, test for current fold
    #PUT HERE SOME FUNCTION TO SPLIT BETTER DATASET
    dummy = np.load(TARGET_LOAD)
    dummy = dummy.item()
    foldable_list = list(dummy.keys())
    fold_actors_list = uf.folds_generator(num_folds, foldable_list, percs)
    train_list = fold_actors_list[int(num_fold)]['train']
    val_list = fold_actors_list[int(num_fold)]['val']
    test_list = fold_actors_list[int(num_fold)]['test']
    del dummy

    #if tensors of current fold has not been computed:
    if not os.path.exists(test_target_path):
        #load merged dataset, compute and save current tensors
        predictors_merged = np.load(PREDICTORS_LOAD)
        target_merged = np.load(TARGET_LOAD)
        predictors_merged = predictors_merged.item()
        target_merged = target_merged.item()

        print ('\n building dataset for current fold')
        print ('\n training:')
        training_predictors, training_target = uf.build_matrix_dataset(predictors_merged,
                                                            target_merged, train_list)
        print ('\n validation:')

        validation_predictors, validation_target = uf.build_matrix_dataset(predictors_merged,
                                                            target_merged, val_list)
        print ('\n test:')
        test_predictors, test_target = uf.build_matrix_dataset(predictors_merged,
                                                            target_merged, test_list)


        np.save(train_pred_path, training_predictors)
        np.save(train_target_path, training_target)
        np.save(val_pred_path, validation_predictors)
        np.save(val_target_path, validation_target)
        np.save(test_pred_path, test_predictors)
        np.save(test_target_path, test_target)

    else:
        #load pre-computed tensors
        training_predictors = np.load(train_pred_path)
        training_target = np.load(train_target_path)
        validation_predictors = np.load(val_pred_path)
        validation_target = np.load(val_target_path)
        test_predictors = np.load(test_pred_path)
        test_target = np.load(test_target_path)

    #normalize to 0 mean and unity std (according to training set mean and std)
    tr_mean = np.mean(training_predictors)
    tr_std = np.std(training_predictors)
    training_predictors = np.subtract(training_predictors, tr_mean)
    training_predictors = np.divide(training_predictors, tr_std)
    validation_predictors = np.subtract(validation_predictors, tr_mean)
    validation_predictors = np.divide(validation_predictors, tr_std)
    test_predictors = np.subtract(test_predictors, tr_mean)
    test_predictors = np.divide(test_predictors, tr_std)

    #OVERFITTING TEST!!! REMOVE THESE LINES FOR PROPER TRAINING
    '''
    validation_predictors = training_predictors.copy()
    validation_target = training_target.copy()
    '''

    #normalize labels between 0 and 1
    '''
    max_labels = [np.max(training_target), np.max(validation_target), np.max(test_target)]
    max_val = float(np.max(max_labels))
    training_target = np.divide(training_target, max_val)
    validation_target = np.divide(validation_target, max_val)
    test_target = np.divide(test_target, max_val)
    '''

    #select a subdataset for testing (to be commented when normally trained)
    '''
    bound = 30
    training_predictors = training_predictors[:bound]
    training_target = training_target[:bound]
    validation_predictors = validation_predictors[:bound]
    validation_target = validation_target[:bound]
    test_predictors = test_predictors[:bound]
    test_target = test_target[:bound]
    '''

    #reshape tensors
    training_predictors = training_predictors.reshape(training_predictors.shape[0], training_predictors.shape[1],training_predictors.shape[2], 1)
    validation_predictors = validation_predictors.reshape(validation_predictors.shape[0], validation_predictors.shape[1], validation_predictors.shape[2], 1)
    test_predictors = test_predictors.reshape(test_predictors.shape[0], test_predictors.shape[1], test_predictors.shape[2], 1)
    time_dim = training_predictors.shape[1]
    features_dim = training_predictors.shape[2]

    #load and compile model (model is in locals()['model'])
    print('\n loading model...')
    model_string = 'model = choose_model.' + architecture + '(time_dim, features_dim, parameters)'
    exec(model_string)
    locals()['model'].compile(loss=loss_function, optimizer=opt, metrics=metrics_list)
    print (locals()['model'].summary())

    #callbacks
    best_model = ModelCheckpoint(SAVE_MODEL, monitor=save_best_model_metric, save_best_only=True, mode=save_best_model_mode)  #save the best model
    early_stopping_monitor = EarlyStopping(patience=patience)  #stops training when the model is not improving
    if early_stopping:
        callbacks_list = [early_stopping_monitor, best_model]
    else:
        callbacks_list = [best_model]

    #run training
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    model_folder = os.path.dirname(SAVE_MODEL)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    history = locals()['model'].fit(training_predictors,training_target, epochs=num_epochs,
                                validation_data=(validation_predictors,validation_target), callbacks=callbacks_list, batch_size=batch_size)

    train_loss_hist = history.history['loss']
    val_loss_hist = history.history['val_loss']
    if task_type != 'regression':
        train_acc_hist = history.history['acc']
        val_acc_hist = history.history['val_acc']

    #compute results on the best saved model
    K.clear_session()  #free GPU
    best_model = load_model(SAVE_MODEL)  #load best saved model

    train_score = best_model.evaluate(training_predictors, training_target)
    val_score = best_model.evaluate(validation_predictors, validation_target)
    test_score = best_model.evaluate(test_predictors, test_target)

    #save results in temp dict file
    temp_results = {}

    #save loss
    temp_results['train_loss'] = train_score[0]
    temp_results['val_loss'] = val_score[0]
    temp_results['test_loss'] = test_score[0]

    #save acc if classification
    if task_type != 'regression':
        temp_results['train_acc'] = train_score[1]
        temp_results['val_acc'] = val_score[1]
        temp_results['test_acc'] = test_score[1]

    #save history
    temp_results['train_loss_hist'] = train_loss_hist
    temp_results['val_loss_hist'] = val_loss_hist
    if task_type != 'regression':
        temp_results['train_acc_hist'] = train_acc_hist
        temp_results['val_acc_hist'] = val_acc_hist

    #save actors present in current fold
    temp_results['training_actors'] = train_list
    temp_results['validation_actors'] = val_list
    temp_results['test_actors'] = test_list

    print (temp_results)

    np.save(results_path, temp_results)

if __name__ == '__main__':
    main()















#
