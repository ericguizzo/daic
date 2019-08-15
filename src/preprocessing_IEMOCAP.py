import utility_functions as uf
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
                'xxx':None}

num_classes_IEMOCAP = 10
wavname = 'Ses01F_impro01_F001.wav'

def get_label_IEMOCAP(wavname):
    '''
    compute one hot label starting from wav filename
    '''
    wavname = wavname.split('/')[-1]
    session = int(wavname.split('_')[0][3:5])
    trans_file = '_'.join(wavname.split('_')[:2]) + '.txt'
    ID = wavname.split('.')[0]
    trans_path = os.path.join(INPUT_IEMOCAP_FOLDER, 'Session' + str(session),
                            'dialog/EmoEvaluation', trans_file)
    print (trans_path)

    #trans_path = '/home/eric/Desktop/Ses01F_impro01.txt'
    with open(trans_path) as f:
        contents = f.readlines()

    str_label = list(filter(lambda x: ID in x, contents))[0].split('\t')[2]
    int_label = label_to_int[str_label]

    if int_label != None:
        output = uf.onehot(int_label, num_classes_IEMOCAP)
    else:
        output = None

    print (output)

    return output

def get_sounds_list(input_folder=INPUT_IEMOCAP_FOLDER):
    contents = os.listdir(input_folder)
    contents = list(filter(lambda x: 'Session' in x, contents))
    print (contents)


get_sounds_list()
#get_label_IEMOCAP(wavname)
