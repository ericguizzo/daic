import utility_functions as uf
import numpy as np
import os, sys
import configparser
import loadconfig


config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

#get values from config file
INPUT_IEMOCAP_FOLDER = cfg.get('preprocessing', 'input_iemocap_folder')
OUTPUT_FOLDER = cfg.get('preprocessing', 'output_folder')

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




sounds_list = get_sounds_list()
filtered_list = filter_labels(sounds_list)
print (len(sounds_list))
print (len(filtered_list))
#get_label_IEMOCAP(wavname)
