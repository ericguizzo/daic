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


def spectrum(x, M=WINDOW_SIZE, N=FFT_SIZE, H=HOP_SIZE_STFT, fs=SR, window_type=WINDOW_TYPE):
    '''
    magnitudes spectrum
    '''
    SP = librosa.core.stft(x, n_fft=N, hop_length=H, window=window_type)
    SP = np.abs(SP)
    SP = np.power(SP, 2./3.)  #power law compression
    SP = np.rot90(SP)

    return SP


def spectrum_CQ(x, H=HOP_SIZE_CQT, fs=SR, bins_per_octave=BINS_PER_OCTAVE, n_bins=N_BINS, fmin=FMIN):
    '''
    magnitudes constant-q transform (log spectrum)
    '''
    CQT = librosa.core.cqt(x, hop_length=H, sr=fs, bins_per_octave=24, n_bins=168, fmin=55)
    CQT = np.abs(CQT)
    CQT = np.power(CQT, 2./3.)  #power law compression
    CQT = np.rot90(CQT)

    return CQT


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
