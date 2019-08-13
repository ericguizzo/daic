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
WINDOW_SIZE = cfg.getint('stft', 'window_size')
FFT_SIZE = cfg.getint('stft', 'fft_size')
HOP_SIZE = cfg.getint('stft', 'hop_size')
WINDOW_TYPE = str(cfg.get('stft', 'window_type'))
SR = cfg.getint('sampling', 'sr_target')

def spectrum(x, M=WINDOW_SIZE, N=FFT_SIZE, H=HOP_SIZE, fs=SR, window_type=WINDOW_TYPE):
    '''
    magnitudes spectrum
    '''
    SP = librosa.core.stft(x, n_fft=N, hop_length=H, window=window_type)
    SP = np.abs(SP)
    SP = np.power(SP, 2./3.)  #power law compression

    return SP


def spectrum_CQ(x, H=HOP_SIZE, fs=SR):
    '''
    magnitudes constant-q transform (log spectrum)
    '''
    CQT = librosa.core.cqt(x, hop_length=H, sr=fs, bins_per_octave=24, n_bins=168, fmin=55)
    CQT = np.abs(CQT)
    CQT = np.power(CQT, 2./3.)  #power law compression

    return CQT

def mfcc(x, M=WINDOW_SIZE, N=FFT_SIZE, H=HOP_SIZE, fs=SR, window_type=WINDOW_TYPE):
    '''
    mel frequency cepstral coefficients
    '''
    MFCC = librosa.feature.mfcc(x, sr=fs, n_mfcc=50)

    return MFCC

def mfcc2(x, M=WINDOW_SIZE, N=FFT_SIZE, H=HOP_SIZE, fs=SR, window_type=WINDOW_TYPE):
	'''
	-extract features from audio file
	-Features:
		MFCC (24 COEFFS)
		SPECTRAL CONTRAST (6 COEFFS)
		PCA (of mfccs)
		HFC
		SPECTRAL CENTROID
		SPECTRAL ENERGY
		F0
		PITCH CONFIDENCE
		output shape = n_data, time, features
		profile = [24, 6, 1, 1, 1, 1, 1, 1]
	'''
	#audioLoader = ess.EasyLoader(filename=file_name, sampleRate=fs)
	#create essentia instances
	x = essentia.array(x)
	spectrum = ess.Spectrum(size=N)
	window = ess.Windowing(size=M, type=window_type)
	mfcc = ess.MFCC(numberCoefficients=24, inputSize=N/2+1)

	#init vectors
	MFCC = []


	#compute features for every stft frame
	for frame in ess.FrameGenerator(x, frameSize=M, hopSize=H, startFromZero=True): #generate frames
		wX = window(frame)  #window frame
		mX = spectrum(wX)  #compute fft

		mfcc_bands, mfcc_coeffs = mfcc(mX)
		MFCC.append(mfcc_coeffs)

	F0, PITCH_CONFIDENCE = pitch_extractor(x)  #estimate pitch in time domain

	#convert into numpy matrices
	MFCC = essentia.array(MFCC).T


	return MFCC
