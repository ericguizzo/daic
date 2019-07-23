from __future__ import print_function
from scipy.signal import iirfilter, butter, filtfilt, lfilter
import essentia.standard as ess
import essentia
import configparser
import librosa
import numpy as np
import os, sys
import pandas
import loadconfig

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

#get values from config file
WINDOW_SIZE = cfg.getint('stft', 'window_size')
FFT_SIZE = cfg.getint('stft', 'fft_size')
HOP_SIZE = cfg.getint('stft', 'hop_size')
WINDOW_TYPE = cfg.get('stft', 'window_type')
SR = cfg.getint('sampling', 'sr_target')
SEQUENCE_LENGTH = cfg.getfloat('preprocessing', 'sequence_length')
SEQUENCE_OVERLAP = cfg.getfloat('preprocessing', 'sequence_overlap')
#in
INPUT_AUDIO_FOLDER =  cfg.get('preprocessing', 'input_audio_folder')
INPUT_LABELS_FOLDER =  cfg.get('preprocessing', 'input_labels_folder')
INPUT_TRANSCRIPTS_FOLDER =  cfg.get('preprocessing', 'input_transcripts_folder')
#out
OUTPUT_PREDICTORS_PATH = cfg.get('preprocessing', 'output_predictors')
OUTPUT_TARGET_PATH = cfg.get('preprocessing', 'output_target')

def preemphasis(input_vector, fs):
    '''
    2 simple high pass FIR filters in cascade to emphasize high frequencies
    and cut unwanted low-frequencies
    '''
    #first gentle high pass
    alpha=0.5
    present = input_vector
    zero = [0]
    past = input_vector[:-1]
    past = np.concatenate([zero,past])
    past = np.multiply(past, alpha)
    filtered1 = np.subtract(present,past)
    #second 30 hz high pass
    fc = 100.  # Cut-off frequency of the filter
    w = fc / (fs / 2.) # Normalize the frequency
    b, a = butter(8, w, 'high')
    output = filtfilt(b, a, filtered1)

    return output

def extract_features(x, M=WINDOW_SIZE, N=FFT_SIZE, H=HOP_SIZE, fs=SR, window_type=WINDOW_TYPE):
    '''
    extract magnitudes spectra from input vector and apply power-law compression
    '''
    #init functions and vectors
    x = essentia.array(x)
    spectrum = ess.Spectrum(size=N)
    window = ess.Windowing(size=M, type=WINDOW_TYPE)
    SP = []

    #compute STFT
    for frame in ess.FrameGenerator(x, frameSize=M, hopSize=H, startFromZero=True): #generate frames
        wX = window(frame)  #window frame
        mX = spectrum(wX)  #compute fft
        SP.append(mX)

    SP = essentia.array(SP)
    SP = np.power(SP, 2./3.)  #power law compression
    SP = SP[:,:int(FFT_SIZE/2+1)]  #cut upper spectrum (above 8 khz)

    return SP

def build_labels_dict(labels_folder):
    #build dict: Participant_ID: label
    dict = {}
    target_column = 'PHQ8_Score'  #target column to get the classification label
    contents = os.listdir(labels_folder)
    for i in contents:
        temp_path = os.path.join(labels_folder, i)
        temp_data = pandas.read_csv(temp_path)
        for index, row in temp_data.iterrows():
            ID = int(row['Participant_ID'])
            label = row[target_column]
            dict[ID] = label
    return dict

def build_bands_dict(labels_dict,n_bands=4):
    #n_bands must be sottomultiplo of label range (24)
    train_perc = 0.7
    val_perc = 0.2
    test_perc = 0.1
    labels = []
    for i in labels_dict:
        labels.append(labels_dict[i])
    #splict maintaining the same amount af datapoints in n_bands
    min_label = np.min(labels)
    max_label = np.max(labels) + 1
    n_elsxband = max_label/n_bands
    bands = np.arange(n_elsxband, max_label, n_elsxband)
    bands = np.append(min_label, bands)
    bands_dict = {}
    output_dict = {'train': [], 'val': [], 'test': []}
    #create dict with items split by band divided in train, val, test
    for i in range(n_bands):
        start = bands[i]
        bands_dict[i] = {}
        end = bands[i] + (n_elsxband-1)
        filt = lambda x: x >= start and x<=end
        tot_band_items = len(list(filter(filt, labels)))
        train_band_items = int(np.floor(tot_band_items*train_perc))
        val_band_items = int(np.ceil(tot_band_items*val_perc))
        test_band_items = int(tot_band_items-(train_band_items+val_band_items))
        bands_dict[i]['tot'] = tot_band_items
        bands_dict[i]['bounds'] = [start, end]
        bands_dict[i]['train'] = train_band_items
        bands_dict[i]['val'] = val_band_items
        bands_dict[i]['test'] = test_band_items
        #print (bands_dict)

    return bands_dict

def build_split_dict(labels_dict, bands_dict, sequence):
    train_dict = {}
    val_dict = {}
    test_dict = {}
    #for every band
    for band in bands_dict:
        tr_count = 0
        val_count = 0
        ts_count = 0
        bounds = bands_dict[band]['bounds']
        tr_n = bands_dict[band]['train']
        val_n = bands_dict[band]['val']
        ts_n = bands_dict[band]['test']
        #for every sample
        for i in sequence:
            label =  labels_dict[i]
            #if sample is within band's bounds
            if label >= bounds[0] and label <= bounds[1]:
                #fill tr, val and ts dicts with correct
                if tr_count < tr_n:
                    train_dict[i] = labels_dict[i]
                    tr_count += 1
                elif val_count < val_n:
                    val_dict[i] = labels_dict[i]
                    val_count += 1
                elif ts_count < ts_n:
                    test_dict[i] = labels_dict[i]
                    ts_count += 1

    print ('Sanity checks...')
    print ('Means:')
    print ('train:' + str(np.mean(list(test_dict.values()))))
    print ('val:' + str(np.mean(list(val_dict.values()))))
    print ('test:' + str(np.mean(list(test_dict.values()))))
    print ('Stds:')
    print ('train:' + str(np.std(list(test_dict.values()))))
    print ('val:' + str(np.std(list(val_dict.values()))))
    print ('test:' + str(np.std(list(test_dict.values()))))

    tot_split = list(train_dict.keys()) + list(val_dict.keys()) + list(test_dict.keys())
    tot_orig = list(labels_dict.keys())
    tot_split = sorted(tot_split)
    tot_orig = sorted(tot_orig)
    equal = tot_split == tot_orig

    print('Is split dataset coherent with original? ' + str(equal))

    return train_dict, val_dict, test_dict


def build_transcripts_dict(transcripts_folder):
    dict = {}
    start_c = 'start_time'  #target column to get the classification label
    end_c = 'stop_time'
    speaker_c = 'speaker'
    text_c = 'value'
    contents = os.listdir(transcripts_folder)
    #contents = contents[:1]
    #for every file get the list of bounds
    for i in contents:
        bounds_participant = []
        text_participant = []
        temp_ID = int(i.split('_')[0])
        dict[temp_ID] = {}
        temp_path = os.path.join(transcripts_folder, i)
        temp_data = pandas.read_csv(temp_path, sep='\t')
        for index, row in temp_data.iterrows():
            speaker = row[speaker_c]
            temp_bounds = [row[start_c], row[end_c]]
            text = row[text_c]
            if speaker == 'Participant':
                bounds_participant.append(temp_bounds)
                text_participant.append(text)
        dict[temp_ID]['bounds'] = bounds_participant
        dict[temp_ID]['text'] = text_participant

    return dict


def cut_sound_file(sound_file, bounds_list, sequence_length, sequence_overlap):
    cuts = []
    sec2samps = lambda x: int(np.round(float(x) * SR))
    dur = sec2samps(sequence_length)
    overlap = sec2samps(sequence_overlap)
    librosa_SR = SR
    if SR == 16000:
        librosa_SR = None
    samples, sr = librosa.core.load(sound_file, sr=librosa_SR)
    for i in bounds_list:
        bounds = [sec2samps(i[0]), sec2samps(i[1])]
        curr_cut = samples[bounds[0]:bounds[1]]
        #zeropad if cut is shorter than sequence length
        if len(curr_cut) < dur:
            pad = np.zeros(dur)
            pad[:len(curr_cut)] = curr_cut
            curr_cut = pad
            cuts.append(curr_cut)
        #segment if cut is longer than dur + overlap
        elif len(curr_cut) > (dur + overlap):
            pointer = np.arange(0, len(curr_cut)-dur, overlap, dtype='int')  #initail positions of segments
            for start in pointer:
                stop = int(start + dur)
                if stop <= len(curr_cut):
                    curr_segment = curr_cut[start:stop]
                    cuts.append(curr_segment)
                else:  #last datapoint has a different overlap
                    curr_segment = features[dur:]
                    cuts.append(curr_segment)
    cuts = np.array(cuts)

    return cuts


def build_preprocessing_dicts(audio_folder, labels_dict, transcripts_dict):
    predictors = {}
    target = {}
    sounds_list = os.listdir(audio_folder)
    #sounds_list = sounds_list[:3]
    num_sounds = len(sounds_list)
    index = 0
    for datapoint in sounds_list:
        participant_ID = int(datapoint.split('_')[0])
        label = labels_dict[participant_ID]
        predictors[participant_ID] = []
        target[participant_ID] = []
        sound_file = os.path.join(audio_folder, datapoint)
        bounds_list = transcripts_dict[participant_ID]['bounds']
        segments = cut_sound_file(sound_file, bounds_list, SEQUENCE_LENGTH, SEQUENCE_OVERLAP)
        for cut in segments:
            feats = preemphasis(cut, SR) #apply preemphasis
            feats = extract_features(feats)  #extract features
            predictors[participant_ID].append(feats)
            target[participant_ID].append(label)
        index += 1

        perc = int(index / num_sounds * 20)
        perc_progress = int(np.round((float(index)/num_sounds) * 100))
        inv_perc = int(20 - perc - 1)
        strings = '[' + '=' * perc + '>' + '.' * inv_perc + ']' + ' Progress: ' + str(perc_progress) + '%'
        print ('\r', strings, end='')

    return predictors, target


def main():
    labels_dict = build_labels_dict(INPUT_LABELS_FOLDER)
    transcripts_dict = build_transcripts_dict(INPUT_TRANSCRIPTS_FOLDER)
    #bands_dict = build_bands_dict(labels_dict, n_bands=4)
    #sequence = labels_dict.keys()  #ROTATE THIS TO OBTAIN XVALIDATION
    #train_dict, val_dict, test_dict = build_split_dict(labels_dict, bands_dict, sequence)
    predictors, target = build_preprocessing_dicts(INPUT_AUDIO_FOLDER, labels_dict, transcripts_dict)
    np.save(OUTPUT_PREDICTORS_PATH, predictors)
    np.save(OUTPUT_TARGET_PATH, target)
    print ('\n Succesfully saved matrices')


if __name__ == '__main__':
    main()
