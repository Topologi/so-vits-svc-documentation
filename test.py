import os.path

import torch

import hubert.encode
import preprocess_wave

if __name__ == '__main__':
    hubert_net = hubert.encode.get_hubert_soft_encoder(proxy={'http': 'http://localhost:7891',
                                                              'https': 'http://localhost:7891'})
    datasets = preprocess_wave.load_hubert_audio('./samples', './samples/output_hubert')
    hubert_net.cuda()
    for data, filename, sound_folder, hubert_folder in datasets:
        torch.save(hubert_net.units(data.cuda()), os.path.join(hubert_folder, f'{filename}.npy'))

