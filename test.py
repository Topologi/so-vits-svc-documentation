import os.path

import librosa
import torch
import torchaudio

import hubert.encode
import preprocess_wave

if __name__ == '__main__':
    t = librosa.load('./datasets/sounds/1/vocals0_13.wav')
    print(t[0].shape)

