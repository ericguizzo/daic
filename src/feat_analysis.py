import numpy as np
import essentia.standard as ess
import essentia
import essentia
import configparser
import loadconfig
import utilities_func as uf

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

#get values from config file
WINDOW_SIZE = cfg.getint('stft_toydata', 'window_size')
FFT_SIZE = cfg.getint('stft_toydata', 'fft_size')
HOP_SIZE = cfg.getint('stft_toydata', 'hop_size')
WINDOW_TYPE = str(cfg.get('stft_toydata', 'window_type'))
SR = cfg.getint('sampling_toydata', 'sr_tess')
fps = 25  #annotations per second
hop_annotation = SR /fps
frames_per_annotation = hop_annotation/float(HOP_SIZE)

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
        ###############################OPTIMIZATION[[[[[[[[[[[[[[]]]]]]]]]]]]]]
        #DEPRECATED
        #################################################
        SP.append(mX)

    SP = essentia.array(SP)
    SP = np.power(SP, 2./3.)  #power law compression
    SP = SP[:,:int(FFT_SIZE/4+1)]

    return SP
