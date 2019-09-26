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
INPUT_GSC_FOLDER = cfg.get('preprocessing', 'input_audio_folder_gsc')
OUTPUT_FOLDER = cfg.get('preprocessing', 'output_folder')
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
if AUGMENTATION:
    print ('Augmentation: ' + str(AUGMENTATION) + ' | num_aug_samples: ' + str(NUM_AUG_SAMPLES) )
else:
    print ('Augmentation: ' + str(AUGMENTATION))

print ('Segmentation: ' + str(SEGMENTATION))
print ('Features type: ' + str(FEATURES_TYPE))


print ('loading data...')
actors_list = []
sounds_list = []  #contains all sound paths
labels = os.listdir(INPUT_GSC_FOLDER)
if '.DS_Store' in labels:
    labels.remove('.DS_Store')
for label in labels:
    path = INPUT_GSC_FOLDER + '/' + label
    sounds = os.listdir(path)
    for sound in sounds:
        sound_path = label + '/' + sound
        actor_id = sound.split('_')[0]
        if actor_id not in actors_list:
            actors_list.append(actor_id)  #unique items in the list
        sounds_list.append(sound_path)  #append path to sound list
actors_dict = {}
for i in range(len(actors_list)):
    actors_dict[i] = actors_list[i]
labels_dict = {}
for i in range(len(labels)):
    labels_dict[labels[i]] = i
assoc_dict = {}
#associates to actor NUMERIC ID all sounds (paths) he recorded
for i in actors_dict.keys():
    actor_id = actors_dict[i]
    find_his_sounds = lambda x: actor_id in x
    his_sounds = list(filter(find_his_sounds, sounds_list))
    assoc_dict[i] = his_sounds

num_classes_speechCmd = len(labels)
num_actors = len(actors_list)


def get_max_length_GSC(input_folder=INPUT_GSC_FOLDER):
    '''
    get longest audio file (insamples) for eventual zeropadding
    '''
    max_file_length, sr = uf.find_longest_audio(input_folder + '/bed')
    #max_file_length, sr = uf.find_longest_audio_list(input_list)
    max_file_length = int(max_file_length * sr)

    return max_file_length


def get_label_GSC(wavname):
    '''
    compute one hot label starting from wav filename
    '''

    label = wavname.split('/')[-2]  #string_label
    int_label = labels_dict[label]
    one_hot_label = (uf.onehot(int(int_label), num_classes_speechCmd))

    return one_hot_label


def main():
    '''
    custom preprocessing routine for the iemocap dataset
    '''
    print ('')
    print ('Setting up preprocessing...')
    print('')
    if SEGMENTATION:
        max_file_length = 1
    else:
        max_file_length=get_max_length_GSC(INPUT_GSC_FOLDER)  #get longest file in samples

    #init predictors and target dicts
    predictors = {}
    target = {}
    #create output paths for the npy matrices
    appendix = '_' + FEATURES_TYPE

    if AUGMENTATION:
        predictors_save_path = os.path.join(OUTPUT_FOLDER, 'gsc' + appendix + '_aug' + str(NUM_AUG_SAMPLES) + '_predictors.npy')
        target_save_path = os.path.join(OUTPUT_FOLDER, 'gsc' + appendix + '_aug' + str(NUM_AUG_SAMPLES) + '_target.npy')
    else:
        predictors_save_path = os.path.join(OUTPUT_FOLDER, 'gsc' + appendix + '_predictors.npy')
        target_save_path = os.path.join(OUTPUT_FOLDER, 'gsc' + appendix + '_target.npy')
    index = 1  #index for progress bar
    for i in range(num_actors):
        #print progress bar
        #uf.print_bar(index, num_actors)
        #get foldable item
        curr_list = assoc_dict[i]
        curr_list = [os.path.join(INPUT_GSC_FOLDER, x) for x in curr_list]
        fold_string = '\nPreprocessing foldable item: ' + str(index) + '/' + str(num_actors)
        print (fold_string)
        #preprocess all sounds of the current actor
        #args:1. listof soundpaths of current actor, 2. max file length, 3. function to extract label from filepath

        try:
            curr_predictors, curr_target = pre.preprocess_foldable_item(curr_list, max_file_length, get_label_GSC, True)
            #append preprocessed predictors and target to the dict
            predictors[i] = curr_predictors
            target[i] = curr_target
        except Exception as e:
            print ('')
            print (e)  #PROBABLY SOME FILES ARE CORRUPTED

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
