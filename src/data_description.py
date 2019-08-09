import numpy as np
import essentia.standard as ess
from sklearn.decomposition import PCA
import utilities_func as uf
import essentia
import ConfigParser
import loadconfig

config = loadconfig.load()
cfg = ConfigParser.ConfigParser()
cfg.read(config)

'''
features to discriminate:
-F0  # base pitch
-loud_factor = energy * (spectral_centroid - F0)  #how many harmonics = how much speaker is yelling
-onsets = F0 confidence #pattern of spoken words
#probably multiply all by rounded onsets
'''


#get values from config file
WINDOW_SIZE = cfg.getint('stft', 'window_size')
FFT_SIZE = cfg.getint('stft', 'fft_size')
HOP_SIZE = cfg.getint('stft', 'hop_size')
WINDOW_TYPE = cfg.get('stft', 'window_type')
SR = cfg.getint('sampling', 'sr_target')

def extract_features(x, M=WINDOW_SIZE, N=FFT_SIZE, H=HOP_SIZE, fs=SR, window_type=WINDOW_TYPE):
	'''
	-extract features from audio file
	-Features:
		HFC
		SPECTRAL CENTROID
		SPECTRAL ENERGY
		F0
        loud_factor = energy * (spectral_centroid - F0)  #how many harmonics = how much speaker is yelling
		PITCH CONFIDENCE
	'''
	#audioLoader = ess.EasyLoader(filename=file_name, sampleRate=fs)
	#create essentia instances
	x = essentia.array(x)
	spectrum = ess.Spectrum(size=N)
	window = ess.Windowing(size=M, type=window_type)
	mfcc = ess.MFCC(numberCoefficients=24, inputSize=N/2+1)
	contrast = ess.SpectralContrast(frameSize=M, sampleRate=fs, numberBands = 6)
	hfc = ess.HFC(sampleRate=fs)
	spectralCentroid = ess.SpectralCentroidTime(sampleRate=fs)
	energy = ess.Energy()
	pitch_extractor = ess.PredominantPitchMelodia(frameSize=M, hopSize=H, maxFrequency=1200)
	pca = PCA(n_components=1)
	#init vectors
	MFCC = []
	CONTRAST = []
	HFC = []
	CENTROID = []
	ENERGY = []

	#compute features for every stft frame
	for frame in ess.FrameGenerator(x, frameSize=M, hopSize=H, startFromZero=True): #generate frames
		wX = window(frame)  #window frame
		mX = spectrum(wX)  #compute fft

		mfcc_bands, mfcc_coeffs = mfcc(mX)
		MFCC.append(mfcc_coeffs)

		frame_contrast, frame_valley = contrast(mX)
		CONTRAST.append(frame_contrast)

		frame_hfc = hfc(mX)
		HFC.append(frame_hfc)

		frame_centroid = spectralCentroid(wX) #compute spectral centroid in time domain
		CENTROID.append(frame_centroid)

		frame_energy = energy(mX) #compute spectral energy in time domain
		ENERGY.append(frame_energy)


	F0, PITCH_CONFIDENCE = pitch_extractor(x)  #estimate pitch in time domain

	#convert into numpy matrices
	MFCC = essentia.array(MFCC).T
	CONTRAST = essentia.array(CONTRAST).T
	HFC = essentia.array(HFC)
	CENTROID = essentia.array(CENTROID)
	ENERGY = essentia.array(ENERGY)
	F0 = essentia.array(F0)
	PITCH_CONFIDENCE = essentia.array(PITCH_CONFIDENCE)
	F0 = F0[:len(CENTROID)]
	PITCH_CONFIDENCE = PITCH_CONFIDENCE[:len(CENTROID)]

	#build mfcc pca vector
	mfcc_copy = MFCC.copy().T
	pca.fit(mfcc_copy)
	PCA_VEC = pca.transform(mfcc_copy)
	PCA_VEC = np.reshape(PCA_VEC, PCA_VEC.shape[0])

	features = np.zeros(len(CENTROID)) #init features map
	#construct features matrix
	features = np.vstack((features,MFCC))
	features = np.vstack((features,CONTRAST))
	features = np.vstack((features, PCA_VEC))
	features = np.vstack((features,HFC))
	features = np.vstack((features,CENTROID))
	features = np.vstack((features,ENERGY))
	features = np.vstack((features,F0))
	features = np.vstack((features,PITCH_CONFIDENCE))
	features = np.delete(features, (0), axis=0)  #	delete first empty

	features = np.rot90(features)

	return features
