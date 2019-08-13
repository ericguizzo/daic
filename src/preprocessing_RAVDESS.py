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
print ('Segmentation: ' + str(SEGMENTATION))


num_classes_ravdess = 8
max_file_length, sr = uf.find_longest_audio(INPUT_RAVDESS_FOLDER)
max_file_length = int(max_file_length * SR)
if sr == SR:
    librosa_SR = None
else:
    librosa_SR = SR

def filter_data(contents, criterion, train_list, val_list, test_list):
    '''
    split train, val and test data accodring to criterion
    '''
    actor = lambda x: int(x.split('/')[-1].split('.')[0].split('-')[-1])
    statement = lambda x: int(x.split('/')[-1].split('.')[0].split('-')[-3])
    intensity = lambda x: int(x.split('/')[-1].split('.')[0].split('-')[-4])
    gender = lambda x: 1 if int(x.split('/')[-1].split('.')[0].split('-')[-1]) % 2. == 0 else 0

    if criterion == 'actor':
        filtered_train = list(filter(lambda x: actor(x) in train_list, contents))
        filtered_val = list(filter(lambda x: actor(x) in val_list, contents))
        filtered_test = list(filter(lambda x: actor(x) in test_list, contents))
    elif criterion == 'statement':
        filtered_train = list(filter(lambda x: statement(x) in train_list, contents))
        filtered_val = list(filter(lambda x: statement(x) in val_list, contents))
        filtered_test = list(filter(lambda x: statement(x) in test_list, contents))
    elif criterion == 'intensity':
        filtered_train = list(filter(lambda x: intensity(x) in train_list, contents))
        filtered_val = list(filter(lambda x: intensity(x) in val_list, contents))
        filtered_test = list(filter(lambda x: intensity(x) in test_list, contents))
    elif criterion == 'gender':
        filtered_train = list(filter(lambda x: gender(x) in train_list, contents))
        filtered_val = list(filter(lambda x: gender(x) in val_list, contents))
        filtered_test = list(filter(lambda x: gender(x) in test_list, contents))

    return filtered_train, filtered_val, filtered_test

def preprocess_datapoint(input_sound):
    '''
    generate predictors (stft) and target (valence sequence)
    of one sound file from the OMG dataset
    '''
    raw_samples, sr = librosa.core.load(input_sound, sr=librosa_SR)  #read audio
    if SEGMENTATION:
        # if segment cut initial and final silence if present
        samples = uf.strip_silence(raw_samples)

    else:
        #if not, zero pad all sounds to the same length
        samples = np.zeros(max_file_length)
        samples[:len(raw_samples)] = raw_samples  #zero padding
    #samples = uf.preemphasis(samples, sr)  #apply preemphasis
    feats = fa.extract_features(samples, FEATURES_TYPE)  #extract features
    #compute one hot label
    label = input_sound.split('/')[-1].split('.')[0].split('-')[2]
    one_hot_label = (uf.onehot(int(label)-1, num_classes_ravdess))
    return feats, one_hot_label

def segment_datapoint(features, label):
    '''
    segment features and annotations of one long audio file
    into smaller matrices of length "sequence_length"
    and overlapped by "sequence_overlap"
    '''
    num_frames = features.shape[0]
    step = SEQUENCE_LENGTH*SEQUENCE_OVERLAP  #segmentation overlap step
    pointer = np.arange(0, num_frames, step, dtype='int')  #initail positions of segments
    predictors = []
    target = []
    #slice arrays and append datapoints to vectors
    if SEGMENTATION:
        for start in pointer:
            stop = int(start + SEQUENCE_LENGTH)
            #print start_annotation, stop_annotation, start_features, stop_features
            if stop <= num_frames:
                temp_predictors = features[start:stop]

                predictors.append(temp_predictors)
                target.append(label)
            else:  #last datapoint has a different overlap
                temp_predictors = features[-int(SEQUENCE_LENGTH):]
                predictors.append(temp_predictors)
                target.append(label)
    else:
        predictors.append(features)
        target.append(label)
    predictors = np.array(predictors)
    target = np.array(target)

    return predictors, target

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

        perc_progress = int((index * 100) / num_sounds)
        index += 1
        progress_string = "processed files: " + str(index) + " over " + str(num_sounds) + "  |  progress: " + str(perc_progress) + "%"
        print ('\r', progress_string, end= ' ')
    print ('\n')
    predictors = np.array(predictors)
    target = np.array(target)
    #predictors = np.concatenate(predictors, axis=0)  #reshape arrays
    #target = np.concatenate(target, axis=0)

    #scramble datapoints order
    shuffled_predictors = []
    shuffled_target = []
    num_datapoints = target.shape[0]
    random_indices = np.arange(num_datapoints)
    np.random.shuffle(random_indices)
    for i in random_indices:
        shuffled_predictors.append(predictors[i])
        shuffled_target.append(target[i])
    shuffled_predictors = np.array(shuffled_predictors)
    shuffled_target = np.array(shuffled_target)

    return shuffled_predictors, shuffled_target



def merged_preprocessing():
    criterion = 'actor'
    ac_list = list(range(25))
    contents = os.listdir(INPUT_RAVDESS_FOLDER)
    predictors = {}
    target = {}
    appendix = '_' + FEATURES_TYPE
    predictors_save_path = os.path.join(OUTPUT_FOLDER, 'ravdess_predictors' + appendix + '.npy')
    target_save_path = os.path.join(OUTPUT_FOLDER, 'ravdess_target' + appendix + '.npy')
    for i in ac_list:
        curr_list, dummy, dummy2 = filter_data(contents, criterion, [i+1], [i+1], [i+1])
        curr_predictors, curr_target = preprocess_dataset(curr_list)
        predictors[i] = curr_predictors
        target[i] = curr_target
    np.save(predictors_save_path, predictors)
    np.save(target_save_path, target)

if __name__ == '__main__':
    merged_preprocessing()
