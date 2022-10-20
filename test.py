import os
import hubert.encode
import torchaudio
import librosa

if __name__ == '__main__':
    audio, sr = torchaudio.load('./samples/sample_out1.wav')
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = audio.unsqueeze(0).cuda()
    hubert_net = hubert.encode.get_hubert_soft_encoder(proxy={'http': 'http://localhost:7891'})
    units = hubert_net.units(audio)
    print(units.shape)

