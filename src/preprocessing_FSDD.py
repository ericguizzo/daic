import utility_functions as uf
import preprocessing_utils as pre
import numpy as np
import os, sys
import configparser
import loadconfig


config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

#get values from config file
FEATURES_TYPE = cfg.get('feature_extraction', 'features_type')
SR = cfg.getint('sampling', 'sr_target')
AUGMENTATION = eval(cfg.get('feature_extraction', 'augmentation'))
NUM_AUG_SAMPLES = eval(cfg.get('feature_extraction', 'num_aug_samples'))
SEGMENTATION = eval(cfg.get('feature_extraction', 'segmentation'))
INPUT_FSDD_FOLDER = cfg.get('preprocessing', 'input_fsdd_folder')
OUTPUT_FOLDER = cfg.get('preprocessing', 'output_folder')
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
if AUGMENTATION:
    print ('Augmentation: ' + str(AUGMENTATION) + ' | num_aug_samples: ' + str(NUM_AUG_SAMPLES) )
else:
    print ('Augmentation: ' + str(AUGMENTATION))

print ('Segmentation: ' + str(SEGMENTATION))
print ('Features type: ' + str(FEATURES_TYPE))


assoc_dict = {0: 'jackson',
              1: 'nicolas',
              2: 'theo',
              3: 'yweweler'}

num_classes_FSDD = 10


def get_max_length_FSDD (input_folder):
    '''
    get longest audio file (insamples) for eventual zeropadding
    '''
    max_file_length, sr = uf.find_longest_audio(input_folder)
    max_file_length = int(max_file_length * sr)

    return max_file_length

def get_label_FSDD(wavname):
    '''
    compute one hot label starting from wav filename
    '''
    label = wavname.split('/')[-1].split('.')[0].split('_')[0]
    one_hot_label = (uf.onehot(int(label), num_classes_FSDD))

    return one_hot_label


def filter_actor_list(sounds_list, actor):
    filtered_list = list(filter(lambda x: actor in x, sounds_list))
    return filtered_list

def main():
    '''
    custom preprocessing routine for the iemocap dataset
    '''
    print ('')
    print ('Setting up preprocessing...')
    print('')
    ac_list = list(range(4))
    contents = os.listdir(INPUT_FSDD_FOLDER)
    contents = [os.path.join(INPUT_FSDD_FOLDER, x) for x in contents]  #complete paths
    if SEGMENTATION:
        max_file_length = 1
    else:
        max_file_length=get_max_length_FSDD(INPUT_FSDD_FOLDER)  #get longest file in samples

    num_files = len(ac_list)
    #init predictors and target dicts
    predictors = {}
    target = {}
    #create output paths for the npy matrices
    appendix = '_' + FEATURES_TYPE
    if AUGMENTATION:
        predictors_save_path = os.path.join(OUTPUT_FOLDER, 'fsdd' + appendix + '_aug' + str(NUM_AUG_SAMPLES) + '_predictors.npy')
        target_save_path = os.path.join(OUTPUT_FOLDER, 'fsdd' + appendix + '_aug' + str(NUM_AUG_SAMPLES) + '_target.npy')
    else:
        predictors_save_path = os.path.join(OUTPUT_FOLDER, 'fsdd' + appendix + '_predictors.npy')
        target_save_path = os.path.join(OUTPUT_FOLDER, 'fsdd' + appendix + '_target.npy')

    index = 1  #index for progress bar
    for i in ac_list:
        #print progress bar
        #uf.print_bar(index, num_files)
        #get foldable item DIVIDING BY ACTORS
        curr_list = filter_actor_list(contents, assoc_dict[i])
        fold_string = '\nPreprocessing foldable item: ' + str(index) + '/' + str(num_files)
        print (fold_string)

        #preprocess all sounds of the current actor
        #args:1. listof soundpaths of current actor, 2. max file length, 3. function to extract label from filepath

        curr_predictors, curr_target = pre.preprocess_foldable_item(curr_list, max_file_length, get_label_FSDD, True)
        #append preprocessed predictors and target to the dict
        predictors[i] = curr_predictors
        target[i] = curr_target

        index +=1
    #save dicts
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
