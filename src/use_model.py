
import configparser
import json
import keras
from keras.models import load_model
import librosa
import numpy as np
import preprocessing_utils as pre
import utility_functions as uf
import matplotlib.pyplot as plt

model_path = '/home/eric/Desktop/all_experiments/modelli/best_model_iemocap_augmentation'
MODEL = load_model(model_path)

labels_iemocap = {0: 'neutral',
                1: 'angry',
                2: 'happy/excited',
                3: 'sad'}

def gen_prediction(sound_path, model=MODEL, preprocessing_type='cqt', normalization=True,
                    sequence_length=4, sr=16000):
    dur = int(sequence_length * sr)
    #load
    samples, sr = librosa.core.load(sound_path, sr=sr)
    samples = uf.strip_silence(samples)
    #norm
    if normalization:
        samples = np.divide(samples, np.max(samples))
        samples = np.multiply(samples, 0.8)
    #zeropad
    if len(samples) <= dur:
        pad = np.zeros(dur)
        pad[:len(samples)] = samples
        samples = pad
    else:
        samples = samples[:dur]
    #feature extraction
    features = pre.extract_features(samples, preprocessing_type)
    features = features.reshape(1, features.shape[0],features.shape[1], 1)
    prediction = model.predict(features)

    print ('\nRaw prediction: ' + str(prediction[0]))
    print ('Predicted class: ' + str(np.argmax(prediction[0])))
    print ('IEMOCAP label: ' + labels_iemocap[np.argmax(prediction[0])])



if __name__ == '__main__':
    gen_prediction('/home/eric/Desktop/alpha_ai/omg/dataset/ravdess/ravdess_test/03-01-01-01-01-01-22.wav')















#
