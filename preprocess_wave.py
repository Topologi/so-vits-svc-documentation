import os
import librosa
import pyworld
import utils
import numpy as np
from scipy.io import wavfile
import argparse


class FeatureInput(object):
    def __init__(self, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, path):
        x, sr = librosa.load(path, sr=self.fs)
        assert sr == self.fs
        f0, t = pyworld.dio(
            x.astype(np.double),
            fs=sr,
            f0_ceil=800,
            frame_period=1000 * self.hop / sr,
        )
        f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)
        for index, pitch in enumerate(f0):
            f0[index] = round(pitch, 1)
        return f0

    # for numpy # code from diffsinger
    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
                self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        # use 0 or 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(np.int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    # for tensor # code from diffsinger
    def coarse_f0_ts(self, f0):
        f0_mel = 1127 * (1 + f0 / 700).log()
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
                self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        # use 0 or 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = (f0_mel + 0.5).long()
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    def save_wav(self, wav, path):
        wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
        wavfile.write(path, self.fs, wav.astype(np.int16))


def get_hparams():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sounds', type=str, default='./datasets/sounds',
                        help='Input datasets sounds wav folder')
    parser.add_argument('-f', '--f0', type=str, default='./datasets/f0',
                        help='Set where to put output f0 data')
    parser.add_argument('-u', '--units', type=str, default='./datasets/speech_units',
                        help='Set where to put output HuBERT units')
    parser.add_argument('-c', '--config', type=str, default="./configs/nyarumul.json",
                        help='JSON file for configuration')
    parser.add_argument('-d', '--description', type=str, default="./datasets/nyarumul.txt",
                        help='TXT file for train data description')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_hparams()
    wavPath = args.sounds
    outF0 = args.f0
    outSpeechUnits = args.units
    if not os.path.exists(wavPath):
        os.makedirs(wavPath)
    if not os.path.exists(outF0):
        os.makedirs(outF0)
    if not os.path.exists(outSpeechUnits):
        os.makedirs(outSpeechUnits)

    # define model and load checkpoint
    hps = utils.get_hparams_from_file(args.config)
    featureInput = FeatureInput(hps.data.sampling_rate, hps.data.hop_length)
    with open(args.description, "w", encoding="utf-8") as vits_train_data_desc:
        for spks in os.listdir(wavPath):
            if os.path.isdir(os.path.join(wavPath, spks)):
                os.makedirs(os.path.join(outF0, spks))
                for file in os.listdir(os.path.join(wavPath, spks)):
                    if file.endswith(".wav"):
                        # 消除文件后缀名
                        file = file[:-4]
                        audio_path = os.path.join(wavPath, spks, f'{file}.wav')
                        feature_pit = featureInput.compute_f0(audio_path)
                        coarse_pit = featureInput.coarse_f0(feature_pit)

                        # TODO: Add HuBERT Generation here

                        np.save(
                            os.path.join(outF0, spks, f'{file}_pitch.npy'),
                            coarse_pit,
                            allow_pickle=False,
                        )

                        np.save(
                            os.path.join(wavPath, spks, f'{file}_nsff0.npy'),
                            feature_pit,
                            allow_pickle=False,
                        )

                        # HuBERT code
                        path_label = os.path.join(outF0, spks, f'{file}.npy')
                        path_pitch = os.path.join(outF0, spks, f'{file}_pitch.npy')
                        path_nsff0 = os.path.join(wavPath, spks, f'{file}_nsff0.npy')
                        print(
                            f"{audio_path}|{spks}|{path_label}|{path_pitch}|{path_nsff0}",
                            file=vits_train_data_desc,
                        )
