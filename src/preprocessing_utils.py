from __future__ import print_function
import numpy as np
import essentia.standard as ess
from scipy.io.wavfile import read, write
import librosa
import essentia
import essentia
import configparser
import loadconfig
import utility_functions as uf

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

#get values from config file
#global
SR = cfg.getint('sampling', 'sr_target')
COMPRESSION = eval(cfg.get('feature_extraction', 'power_law_compression'))
FEATURES_TYPE = cfg.get('feature_extraction', 'features_type')
SEQUENCE_LENGTH = cfg.getfloat('feature_extraction', 'sequence_length')
SEQUENCE_OVERLAP = cfg.getfloat('feature_extraction', 'sequence_overlap')
SEGMENTATION = eval(cfg.get('feature_extraction', 'segmentation'))
#spectrum
WINDOW_SIZE = cfg.getint('feature_extraction', 'window_size')
FFT_SIZE = cfg.getint('feature_extraction', 'fft_size')
HOP_SIZE_STFT = cfg.getint('feature_extraction', 'hop_size_stft')
WINDOW_TYPE = str(cfg.get('feature_extraction', 'window_type'))
#cqt
HOP_SIZE_CQT = cfg.getint('feature_extraction', 'hop_size_cqt')
BINS_PER_OCTAVE = cfg.getint('feature_extraction', 'bins_per_octave')
N_BINS = cfg.getint('feature_extraction', 'n_bins')
FMIN = cfg.getint('feature_extraction', 'fmin')
#mfcc
WINDOW_SIZE_MFCC = cfg.getint('feature_extraction', 'window_size_mfcc')
FFT_SIZE_MFCC = cfg.getint('feature_extraction', 'fft_size_mfcc')
HOP_SIZE_MFCC = cfg.getint('feature_extraction', 'hop_size_mfcc')
WINDOW_TYPE_MFCC = str(cfg.get('feature_extraction', 'window_type_mfcc'))
N_MFCC = cfg.getint('feature_extraction', 'n_mfcc')
#melspectrogram
HOP_SIZE_MEL = cfg.getint('feature_extraction', 'hop_size_mel')
FFT_SIZE_MEL = cfg.getint('feature_extraction', 'fft_size_mel')


def spectrum(x, M=WINDOW_SIZE, N=FFT_SIZE, H=HOP_SIZE_STFT, fs=SR, window_type=WINDOW_TYPE, compression=COMPRESSION):
    '''
    magnitudes spectrum
    '''
    SP = librosa.core.stft(x, n_fft=N, hop_length=H, window=window_type)
    SP = np.abs(SP)
    if compression:
        SP = np.power(SP, 2./3.)  #power law compression
    SP = np.rot90(SP)

    return SP


def spectrum_CQ(x, H=HOP_SIZE_CQT, fs=SR, bins_per_octave=BINS_PER_OCTAVE, n_bins=N_BINS, fmin=FMIN, compression=COMPRESSION):
    '''
    magnitudes constant-q transform (log spectrum)
    '''
    CQT = librosa.core.cqt(x, hop_length=H, sr=fs, bins_per_octave=24, n_bins=168, fmin=55)
    CQT = np.abs(CQT)
    if compression:
        CQT = np.power(CQT, 2./3.)  #power law compression
    CQT = np.rot90(CQT)

    return CQT

def spectrum_mel(x, H=HOP_SIZE_MEL, fs=SR, N=FFT_SIZE_MEL, compression=COMPRESSION):
    '''
    magnitudes constant-q transform (log spectrum)
    '''
    MEL = librosa.feature.melspectrogram(x, sr=fs, n_fft=N, hop_length=H)
    MEL = np.abs(MEL)
    if compression:
        MEL = np.power(MEL, 2./3.)  #power law compression
    MEL = np.rot90(MEL)

    return MEL


def mfcc(x, M=WINDOW_SIZE_MFCC, N=FFT_SIZE_MFCC, H=HOP_SIZE_MFCC, fs=SR,
            window_type=WINDOW_TYPE_MFCC, n_mfcc=N_MFCC):
	'''
	-extract features from audio file
	-Features:
		MFCC (24 COEFFS)

	'''
	#audioLoader = ess.EasyLoader(filename=file_name, sampleRate=fs)
	#create essentia instances
	x = essentia.array(x)
	spectrum = ess.Spectrum(size=N)
	window = ess.Windowing(size=M, type=window_type)
	mfcc = ess.MFCC(numberCoefficients=n_mfcc, inputSize=int(N/2+1), sampleRate=fs, highFrequencyBound=int(fs/2-1))

	#init vectors
	MFCC = []


	#compute features for every stft frame
	for frame in ess.FrameGenerator(x, frameSize=M, hopSize=H, startFromZero=True): #generate frames
		wX = window(frame)  #window frame
		mX = spectrum(wX)  #compute fft

		mfcc_bands, mfcc_coeffs = mfcc(mX)
		MFCC.append(mfcc_coeffs)


	#convert into numpy matrices
	MFCC = essentia.array(MFCC)

	return MFCC

def extract_features(input_vector, features_type):
    if features_type == 'stft':
        feats = spectrum(input_vector)
    elif features_type == 'cqt':
        feats = spectrum_CQ(input_vector)
    elif features_type == 'mel':
        feats = spectrum_mel(input_vector)
    elif features_type == 'mfcc':
        feats = mfcc(input_vector)
    else:
        raise ValueError('Wrong features_type. Possible values: stft, cqt, mfcc')

    return feats


def preprocess_datapoint(input_filename, max_file_length, librosa_SR, hop_size):
    '''
    generate predictors (stft) and target (valence sequence)
    of one sound file from the OMG dataset
    '''
    raw_samples, sr = librosa.core.load(input_filename, sr=librosa_SR)  #read audio
    #audioloader = ess.EasyLoader(sampleRate=SR)
    #raw_samples = audioloader(input_filename)
    if SEGMENTATION:
        seq_len_samps = int(SEQUENCE_LENGTH * sr)
        #librosa does not compute last frame (zeropadding) if
        #len(samples) is not multiple of hop_size
        #librosa is shit
        missing_samples = int(np.ceil(hop_size / seq_len_samps * seq_len_samps))
        pad_length = seq_len_samps + missing_samples
        #pad_length = 64
        pad_length = seq_len_samps
        # if segment cut initial and final silence if present
        #samples = uf.strip_silence(raw_samples)
        if len(raw_samples) < pad_length:
            pad = np.zeros(pad_length)
            pad[:len(raw_samples)] = raw_samples
            samples = pad

    else:
        #if not, zero pad all sounds to the same length
        samples = np.zeros(max_file_length)
        samples[:len(raw_samples)] = raw_samples  #zero padding
    #samples = uf.preemphasis(samples, sr)  #apply preemphasis
    feats = extract_features(samples, FEATURES_TYPE)  #extract features

    return feats

def segment_datapoint(features, label, seq_len_frames):
    '''
    segment features of one long audio file
    into smaller matrices of length "sequence_length"
    and overlapped by "sequence_overlap"
    This function applies the same label to every segmented datapoint!!
    -- label_function is the function that extracts the label
    '''
    num_frames = features.shape[0]
    step = int(np.round(seq_len_frames*SEQUENCE_OVERLAP))  #segmentation overlap step
    pointer = np.arange(0, num_frames, step, dtype='int')  #initail positions of segments
    print ('\npointer')
    print (pointer)
    predictors = []
    target = []
    #slice arrays and append datapoints to vectors
    if SEGMENTATION:
        for start in pointer:
            stop = int(start + seq_len_frames)
            print (start, stop, num_frames)
            if stop <= num_frames:
                temp_predictors = features[start:stop]
                predictors.append(temp_predictors)
                target.append(label)
            else:  #last datapoint has a different overlap
                #temp_predictors = features[-int(seq_len_frames):]
                #predictors.append(temp_predictors)
                #target.append(label)
                pass
    else:
        predictors.append(features)
        target.append(label)
    predictors = np.array(predictors)
    target = np.array(target)
    print ('culo')
    print (predictors.shape)

    return predictors, target

def preprocess_foldable_item(sounds_list, max_file_length, get_label_function):
    '''
    compute predictors and target of all sounds in sound list
    sound_list should contain all filenames of 1 single foldable item
    '''

    predictors = np.array([])
    target = np.array([])

    #compute correct SEQUENCE_LENGTH
    hop_str = 'hop_size = HOP_SIZE_' + FEATURES_TYPE.upper()
    exec(hop_str)

    #librosa sr is None if no resampling is required (speed up)
    '''
    if len(sounds_list) > 1:
        sr, dummy = read(sounds_list[0])
        if sr == SR:
            librosa_SR = None
        else:
            librosa_SR = SR
    else:
        librosa_SR = SR
    '''
    librosa_SR = SR

    #process all files in sound_list
    index = 0
    for sound_file in sounds_list:
        label = get_label_function(sound_file)
        try:
            long_predictors = preprocess_datapoint(sound_file, max_file_length, librosa_SR, locals()['hop_size'])  #compute features
            cut_predictors, cut_target = segment_datapoint(long_predictors, label, locals()['hop_size'])   #segment feature maps
            if not np.isnan(np.std(cut_predictors)):   #some sounds give nan for no reason
                if predictors.shape == (0,):
                    predictors = cut_predictors
                    target = cut_target
                else:
                    predictors = np.append(predictors, cut_predictors, axis=0)
                    target = np.append(target, cut_target, axis=0)
        except (ValueError):
            if str(e) == 'File format b\'FORM\'... not understood.':
                pass

        #uf.print_bar(index, num_sounds)
    predictors = np.array(predictors)
    target = np.array(target)

    return predictors, target

def shuffle_datasets(predictors, target):
    '''
    random shuffle predictors and target matrices
    '''
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
