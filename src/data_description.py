import numpy as np
import essentia.standard as ess
import utility_functions as uf
import sys, os
import essentia
import configparser
import loadconfig
import matplotlib.pyplot as plt

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

in_folder = sys.argv[1]
out_file = sys.argv[2]

#get values from config file
WINDOW_SIZE = cfg.getint('stft', 'window_size')
FFT_SIZE = cfg.getint('stft', 'fft_size')
HOP_SIZE = cfg.getint('stft', 'hop_size')
WINDOW_TYPE = cfg.get('stft', 'window_type')
SR = cfg.getint('sampling', 'sr_target')

def compute_description(x, M=WINDOW_SIZE, N=FFT_SIZE, H=HOP_SIZE, fs=SR, window_type=WINDOW_TYPE):
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
    hfc = ess.HFC(sampleRate=fs)
    spectralCentroid = ess.SpectralCentroidTime(sampleRate=fs)
    energy = ess.Energy()
    pitch_extractor = ess.PredominantPitchMelodia(frameSize=M, hopSize=H, maxFrequency=1200)
    #init vectors
    CONTRAST = []
    HFC = []
    CENTROID = []
    ENERGY = []

    #compute features for every stft frame
    for frame in ess.FrameGenerator(x, frameSize=M, hopSize=H, startFromZero=True): #generate frames
        wX = window(frame)  #window frame
        mX = spectrum(wX)  #compute fft
        frame_hfc = hfc(mX)
        HFC.append(frame_hfc)
        frame_centroid = spectralCentroid(wX) #compute spectral centroid in time domain
        CENTROID.append(frame_centroid)
        frame_energy = energy(mX) #compute spectral energy in time domain
        ENERGY.append(frame_energy)

    F0, SALIENCE = pitch_extractor(x)  #estimate pitch in time domain

    #convert into numpy matrices
    HFC = essentia.array(HFC)
    CENTROID = essentia.array(CENTROID)
    ENERGY = essentia.array(ENERGY)
    F0 = essentia.array(F0)
    SALIENCE = essentia.array(SALIENCE)
    F0 = F0[:len(CENTROID)]
    SALIENCE = SALIENCE[:len(CENTROID)]


    return HFC, CENTROID, ENERGY, F0, SALIENCE

def cut_silence(input_vector, threshold):
    '''
    cut from a signal the segments in which the signal
    is below a certain percentage (threshold) of its excursion
    ex: cut pitch at 0Hz when in reality is silence
    '''
    cut = []
    max = np.max(input_vector)
    min = np.min(input_vector)
    excursion = max - min
    thresh = threshold * (excursion + min)
    for i in input_vector:
        if i >= thresh:
            cut.append(i)
    cut = np.array(cut)

    return cut

def squarify(input_vector):
    '''
    kind of square sigmoid
    '''
    squared = []
    max = np.max(input_vector)
    min = np.min(input_vector)
    mean = (max+min)/2.
    for i in input_vector:
        if i <= mean:
            squared.append(0.)
        else:
            squared.append(1.)
    squared = np.array(squared)

    return squared

def compute_yell_factor(energy, F0, centroid):
    '''
    compute yell factor
    I.E #how many harmonics
    evolution of this parameter describes evolution of timbre

    '''
    yell = []
    for i in range(len(energy)):
        try:
            temp_yell = energy[i] * (centroid[i]-F0[i])

            yell.append(temp_yell)
        except IndexError:
            pass
    yell = np.array(yell)

    return yell


def process_sound(soundfile):
    sr, samples = uf.wavread(soundfile)
    dur = len(samples)
    samples = uf.strip_silence(samples, threshold=31)  #cut initial and final silence
    dur_stripped = len(samples)

    hfc, centroid, energy, F0, salience = compute_description(samples, fs=sr)

    yell_factor = compute_yell_factor(energy, F0, centroid)
    F0 = cut_silence(F0, threshold=0.05)
    salience = squarify(salience)

    mean_hfc = np.mean(hfc)
    std_hfc = np.std(hfc)
    mean_centroid = np.mean(centroid)
    std_centroid = np.std(centroid)
    mean_energy = np.mean(energy)
    std_energy = np.std(energy)
    mean_F0 = np.mean(F0)
    std_F0 = np.std(F0)
    mean_yell_factor = np.mean(yell_factor)
    std_yell_factor = np.std(yell_factor)
    perc_salience = np.mean(salience)

    description = {'mean_hfc': mean_hfc,
                   'std_hfc': std_hfc,
                   'mean_centroid': mean_centroid,
                   'std_centroid': std_centroid,
                   'mean_energy': mean_energy,
                   'std_energy': std_energy,
                   'mean_F0': mean_F0,
                   'std_F0': std_F0,
                   'mean_yell_factor': mean_yell_factor,
                   'std_yell_factor': std_yell_factor,
                   'perc_salience': perc_salience,
                   'dur': dur,
                   'dur_stripped': dur_stripped}

    return description

def main(in_folder):
    contents = os.listdir(in_folder)
    contents = list(filter(lambda x: '.wav' in x, contents))
    dataset_description = {}
    for sound in contents:
        soundfile = os.path.join(in_folder, sound)
        description = process_sound(soundfile)
        dataset_description[sound] = description

    np.save(out_file, dataset_description)





if __name__ == '__main__':
    main(in_folder)
