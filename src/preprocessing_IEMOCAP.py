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
SEGMENTATION = eval(cfg.get('feature_extraction', 'segmentation'))
INPUT_IEMOCAP_FOLDER = cfg.get('preprocessing', 'input_iemocap_folder')
OUTPUT_FOLDER = cfg.get('preprocessing', 'output_folder')

print ('Segmentation: ' + str(SEGMENTATION))
print ('Features type: ' + str(FEATURES_TYPE))

label_to_int_complete = {'neu':0,
                'ang':1,
                'fru':2,
                'hap':3,
                'sad':4,
                'exc':5,
                'fea':6,
                'sur':7,
                'dis':8,
                'xxx':9}

label_to_int = {'neu':0,
                'ang':1,
                'hap':2,
                'exc':2,
                'sad':3,
                'fru':None,
                'fea':None,
                'sur':None,
                'dis':None,
                'oth':None,
                'xxx':None}

num_classes_IEMOCAP = 4
wavname = 'Ses01F_impro01_F001.wav'
#wavname = 'Ses01M_script01_2_F003.wav'

def get_max_length_IEMOCAP(input_list):
    '''
    get longest audio file (insamples) for eventual zeropadding
    '''
    max_file_length, sr = uf.find_longest_audio_list(input_list)
    max_file_length = int(max_file_length * sr)

    return max_file_length

def get_label_IEMOCAP(wavname):
    '''
    compute one hot label starting from wav filename
    '''
    wavname = wavname.split('/')[-1]
    session = int(wavname.split('_')[0][3:5])
    trans_file = '_'.join(wavname.split('_')[:-1]) + '.txt'
    ID = wavname.split('.')[0]
    trans_path = os.path.join(INPUT_IEMOCAP_FOLDER, 'Session' + str(session),
                            'dialog/EmoEvaluation', trans_file)
    #trans_path = '/home/eric/Desktop/Ses01F_impro01.txt'
    with open(trans_path) as f:
        contents = f.readlines()

    str_label = list(filter(lambda x: ID in x, contents))[0].split('\t')[2]
    int_label = label_to_int[str_label]

    if int_label != None:
        output = uf.onehot(int_label, num_classes_IEMOCAP)
    else:
        output = None


    return output

def get_sounds_list(input_folder=INPUT_IEMOCAP_FOLDER):
    '''
    get list of all sound paths in the dataset
    '''
    paths = []
    contents = os.listdir(input_folder)
    contents = list(filter(lambda x: 'Session' in x, contents))
    #iterate sessions
    for session in contents:
        session_path = os.path.join(input_folder, session, 'sentences/wav')
        dialogs = os.listdir(session_path)
        #iterate dialogs
        for dialog in dialogs:
            dialog_path = os.path.join(session_path, dialog)
            utterances = os.listdir(dialog_path)
            #iterate utterance files
            for utterance in utterances:
                utterance_path = os.path.join(dialog_path, utterance)
                paths.append(utterance_path)

    return paths

def filter_labels(sounds_list):
    '''
    filter only sounds with desired labels:
    -neutral
    -happy (excited)
    -angry
    -sad
    '''
    filtered_list = []
    for sound in sounds_list:
        label = get_label_IEMOCAP(sound)
        if type(label) == np.ndarray:
            filtered_list.append(sound)

    return filtered_list

def filter_data_IEMOCAP(sound_file):
    '''
    this function simply returns the input string as a list
    NOTE THAT DOING THIS WE DO NOT SPLIT DATASET ACCORDING TO
    DIFFERENT ACTORS!!!
    we only xfold and tr/val/test split in order to not have segments of the
    same recordings divided in different sets
    '''
    output = [sound_file]

    return output

def main():
    '''
    custom preprocessing routine for the iemocap dataset
    '''
    print ('')
    print ('Setting up preprocessing...')
    print('')
    sounds_list = get_sounds_list(INPUT_IEMOCAP_FOLDER)  #get list of all soundfile paths
    filtered_list = filter_labels(sounds_list)  #filter only sounds of certain labels
    max_file_length=get_max_length_IEMOCAP(filtered_list)  #get longest file in samples
    num_files = len(sounds_list)
    #init predictors and target dicts
    predictors = {}
    target = {}
    #create output paths for the npy matrices
    appendix = '_' + FEATURES_TYPE
    predictors_save_path = os.path.join(OUTPUT_FOLDER, 'iemocap' + appendix + '_predictors.npy')
    target_save_path = os.path.join(OUTPUT_FOLDER, 'iemocap' + appendix + '_target.npy')
    index = 1  #index for progress bar
    for i in filtered_list:
        #print progress bar
        uf.print_bar(index, num_files)
        #get foldable item """NOT DIVIDING BY ACTORS!!!
        curr_list = filter_data_IEMOCAP(i)
        print ('culo')
        print (curr_list)
        #preprocess all sounds of the current actor
        #args:1. listof soundpaths of current actor, 2. max file length, 3. function to extract label from filepath
        curr_predictors, curr_target = pre.preprocess_foldable_item(curr_list, max_file_length, get_label_IEMOCAP)
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
