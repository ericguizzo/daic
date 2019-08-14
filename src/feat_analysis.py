from __future__ import print_function
import numpy as np
import essentia.standard as ess
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
SR = cfg.getint('sampling', 'sr_target')
COMPRESSION = eval(cfg.get('feature_extraction', 'power_law_compression'))
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


def spectrum(x, M=WINDOW_SIZE, N=FFT_SIZE, H=HOP_SIZE_STFT, fs=SR, window_type=WINDOW_TYPE):
    '''
    magnitudes spectrum
    '''
    SP = librosa.core.stft(x, n_fft=N, hop_length=H, window=window_type)
    SP = np.abs(SP)
    if COMPRESSION:
        SP = np.power(SP, 2./3.)  #power law compression
    SP = np.rot90(SP)

    return SP


def spectrum_CQ(x, H=HOP_SIZE_CQT, fs=SR, bins_per_octave=BINS_PER_OCTAVE, n_bins=N_BINS, fmin=FMIN):
    '''
    magnitudes constant-q transform (log spectrum)
    '''
    CQT = librosa.core.cqt(x, hop_length=H, sr=fs, bins_per_octave=24, n_bins=168, fmin=55)
    CQT = np.abs(CQT)
    if COMPRESSION:
        CQT = np.power(CQT, 2./3.)  #power law compression
    CQT = np.rot90(CQT)

    return CQT

def spectrum_mel(x, H=HOP_SIZE_MEL, fs=SR, N=FFT_SIZE_MEL):
    '''
    magnitudes constant-q transform (log spectrum)
    '''
    MEL = librosa.feature.melspectrogram(x, sr=fs, n_fft=N, hop_length=H)
    MEL = np.abs(MEL)
    if COMPRESSION:
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
