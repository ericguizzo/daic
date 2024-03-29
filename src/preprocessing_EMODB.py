from __future__ import print_function
import loadconfig
import configparser
import utility_functions as uf
import preprocessing_utils as pre
import numpy as np
import os, sys

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

#get values from config file
FEATURES_TYPE = cfg.get('feature_extraction', 'features_type')
SR = cfg.getint('sampling', 'sr_target')
SEGMENTATION = eval(cfg.get('feature_extraction', 'segmentation'))
#in
INPUT_EMODB_FOLDER =  cfg.get('preprocessing', 'input_emodb_folder')
#out
OUTPUT_FOLDER = cfg.get('preprocessing', 'output_folder')
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print ('Segmentation: ' + str(SEGMENTATION))
print ('Features type: ' + str(FEATURES_TYPE))


#number of classes of current dataset
num_classes_emodb = 7

assoc_labels = {
                'W': 0,
                'L': 1,
                'E': 2,
                'A': 3,
                'F': 4,
                'T': 5,
                'N': 6
                }

def get_max_length_EMODB(input_folder=INPUT_EMODB_FOLDER, sr=SR):
    '''
    get longest audio file (insamples) for eventual zeropadding
    '''
    max_file_length, sr = uf.find_longest_audio(input_folder)
    max_file_length = int(max_file_length * sr)

    return max_file_length

def get_label_EMODB(input_soundfile, num_classes=num_classes_emodb):
    '''
    compute label starting from soundfile
    '''
    label = input_soundfile.split('/')[-1].split('.')[0][-2]
    int_label = assoc_labels[label]
    one_hot_label = (uf.onehot(int(int_label)-1, num_classes_emodb))

    return one_hot_label


def filter_data_EMODB(contents, item_to_filter):
    '''
    outputs list of soundfiles of the only actor number "item_to_filter"
    '''

    criterion = lambda x: item_to_filter == int(x.split('/')[-1][:2])
    filtered = list(filter(criterion, contents))

    return filtered


def main():
    '''
    custom preprocessing routine for the ravdess dataset
    '''
    print ('')
    print ('Setting up preprocessing...')
    print('')
    #compute max file length of current dataet
    #for the zeropadding
    max_file_length = get_max_length_EMODB()

    #define the list of foldable items. In the case of RAVDESS
    #actors are simply numbered from 0 to 24

    actors_list = [3, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    contents = os.listdir(INPUT_EMODB_FOLDER)  #get list of filepaths
    contents = list(filter(lambda x: '.wav' in x, contents))  #keep only wav files
    contents = [os.path.join(INPUT_EMODB_FOLDER, x) for x in contents]
    #actors_list = actors_list[:2]
    num_files = len(actors_list)
    #init predictors and target dicts
    predictors = {}
    target = {}
    #create output paths for the npy matrices
    appendix = '_' + FEATURES_TYPE
    predictors_save_path = os.path.join(OUTPUT_FOLDER, 'emodb' + appendix + '_predictors.npy')
    target_save_path = os.path.join(OUTPUT_FOLDER, 'emodb' + appendix + '_target.npy')
    #iterate the list of actors
    index = 1  #index for progress bar
    for i in actors_list:
        #print progress bar
        #uf.print_bar(index, num_files)
        #get only soundpaths of current actor
        curr_list = filter_data_EMODB(contents, i)


        fold_string = '\nPreprocessing foldable item: ' + str(index) + '/' + str(num_files)
        print (fold_string)
        #make sure that each item list is a FULL path to a sound file
        #and not only the sound name as os.listdir outputs
        #preprocess all sounds of the current actor
        #args:1. listof soundpaths of current actor, 2. max file length, 3. function to extract label from filepath
        curr_predictors, curr_target = pre.preprocess_foldable_item(curr_list, max_file_length, get_label_EMODB, True)
        #append preprocessed predictors and target to the dict
        predictors[i] = curr_predictors
        target[i] = curr_target

        index +=1
    #save dicts
    #save dicts
    print ('\nSaving matrices...')
    np.save(predictors_save_path, predictors)
    np.save(target_save_path, target)
    #print dimensions
    count = 0
    predictors_dims = 0
    keys = list(predictors.keys())
    for i in keys:
        count += predictors[i].shape[0]
    pred_shape = np.array(predictors[keys[0]]).shape[1:]
    tg_shape = np.array(target[keys[0]]).shape[1:]
    print ('')
    print ('MATRICES SUCCESFULLY COMPUTED')
    print ('')
    print ('Total number of datapoints: ' + str(count))
    print (' Predictors shape: ' + str(pred_shape))
    print (' Target shape: ' + str(tg_shape))


if __name__ == '__main__':
    main()
