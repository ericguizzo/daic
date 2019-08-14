from __future__ import print_function
import loadconfig
import configparser
import utility_functions as uf
from librosa.feature import mfcc
import numpy as np
import librosa
import utility_functions as uf
import os, sys
import matplotlib.pyplot as plt
import feat_analysis as fa
import essentia.standard as ess
import math

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

#get values from config file
FEATURES_TYPE = cfg.get('feature_extraction', 'features_type')
SR = cfg.getint('sampling', 'sr_target')
SEQUENCE_LENGTH = cfg.getfloat('preprocessing', 'sequence_length')
SEQUENCE_OVERLAP = cfg.getfloat('preprocessing', 'sequence_overlap')
#in
INPUT_RAVDESS_FOLDER =  cfg.get('preprocessing', 'input_audio_folder_ravdess')
#out
OUTPUT_FOLDER = cfg.get('preprocessing', 'output_folder')

SEGMENTATION = False

try:
    FEATURES_TYPE = sys.argv[1]
except:
    pass
print ('Segmentation: ' + str(SEGMENTATION))
print ('Features type: ' + str(FEATURES_TYPE))



num_classes_ravdess = 8
max_file_length, sr = uf.find_longest_audio(INPUT_RAVDESS_FOLDER)
max_file_length = int(max_file_length * SR)
if sr == SR:
    librosa_SR = None
else:
    librosa_SR = SR

def extract_label_RAVDESS(input_soundfile):
    #compute one hot label
    label = input_soundfile.split('/')[-1].split('.')[0].split('-')[2]
    one_hot_label = (uf.onehot(int(label)-1, num_classes_ravdess))

    return one_hot_label

def filter_data_RAVDESS(contents, criterion, filter_list):
    '''
    split train, val and test data accodring to criterion
    '''
    actor = lambda x: int(x.split('/')[-1].split('.')[0].split('-')[-1])
    statement = lambda x: int(x.split('/')[-1].split('.')[0].split('-')[-3])
    intensity = lambda x: int(x.split('/')[-1].split('.')[0].split('-')[-4])
    gender = lambda x: 1 if int(x.split('/')[-1].split('.')[0].split('-')[-1]) % 2. == 0 else 0

    if criterion == 'actor':
        filtered = list(filter(lambda x: actor(x) in filter_list, contents))
    elif criterion == 'statement':
        filtered = list(filter(lambda x: statement(x) in filter_list, contents))
    elif criterion == 'intensity':
        filtered = list(filter(lambda x: intensity(x) in filter_list, contents))
    elif criterion == 'gender':
        filtered = list(filter(lambda x: gender(x) in filter_list, contents))

    return filtered



def preprocess_dataset(sounds_list):

    predictors = np.array([])
    target = np.array([])
    num_sounds = len(sounds_list)

    #process all files in folders
    index = 0
    for datapoint in sounds_list:
        sound_file = INPUT_RAVDESS_FOLDER + '/' + datapoint  #get correspective sound
        try:
            long_predictors, long_target = preprocess_datapoint(sound_file)  #compute features
            cut_predictors, cut_target = segment_datapoint(long_predictors, long_target)   #slice feature maps
            if not np.isnan(np.std(cut_predictors)):   #some sounds give nan for no reason
                if predictors.shape == (0,):
                    predictors = cut_predictors
                    target = cut_target
                else:
                    predictors = np.append(predictors, cut_predictors, axis=0)
                    target = np.append(target, cut_target, axis=0)
        except ValueError as e:
            if str(e) == 'File format b\'FORM\'... not understood.':
                pass

        uf.print_bar(index, num_sounds)
    print ('\n')
    predictors = np.array(predictors)
    target = np.array(target)
    #predictors = np.concatenate(predictors, axis=0)  #reshape arrays
    #target = np.concatenate(target, axis=0)





def merged_preprocessing():
    criterion = 'actor'
    ac_list = list(range(25))
    contents = os.listdir(INPUT_RAVDESS_FOLDER)
    predictors = {}
    target = {}
    appendix = '_' + FEATURES_TYPE
    predictors_save_path = os.path.join(OUTPUT_FOLDER, 'ravdess' + appendix + '_predictors.npy')
    target_save_path = os.path.join(OUTPUT_FOLDER, 'ravdess' + appendix + '_target.npy')
    for i in ac_list:
        curr_list, dummy, dummy2 = filter_data(contents, criterion, [i+1], [i+1], [i+1])
        curr_predictors, curr_target = preprocess_dataset(curr_list)
        predictors[i] = curr_predictors
        target[i] = curr_target
    np.save(predictors_save_path, predictors)
    np.save(target_save_path, target)

if __name__ == '__main__':
    merged_preprocessing()
