import argparse
import json
import os
import logging
import utils
from sounds_feature import FeatureInput
from hubert.encode import get_hubert_soft_encoder
import librosa
import torch
from models import SynthesizerTrn
import soundfile as sf


def get_logger():
    fo = '%(asctime)s %(message)s'
    logging.basicConfig(format=fo)
    lo = logging.getLogger('Inference')
    lo.setLevel(logging.DEBUG)
    return lo


def get_units(file_paths) -> []:
    result = []
    hubert_net = get_hubert_soft_encoder()
    for single_file in file_paths:
        sound, sr = librosa.load(single_file, 16000)
        sound = torch.FloatTensor(sound)
        sound = sound.unsqueeze(0).unsqueeze(0)
        with torch.inference_mode():
            units = hubert_net.units(sound)
            result.append(units)
    return result


def get_pitch(file_paths, shift=4, sr=32000, hop_size=320) -> ([], []):
    result = []
    feature_input = FeatureInput(sr, hop_size)
    soft_units = get_units(file_paths)
    for index, single_file in enumerate(file_paths):
        audio, sr = librosa.load(single_file)
        feature_pit = librosa.pyin(audio,
                                   fmin=librosa.note_to_hz('C0'),
                                   fmax=librosa.note_to_hz('C7'),
                                   frame_length=1780, sr=sr)[0]
        feature_pit = utils.resize_2d(feature_pit, soft_units[index].shape[1])
        feature_pit = feature_pit * 2 ** (shift / 12)

        pitch = feature_input.coarse_f0(feature_pit)
        result.append(torch.LongTensor(pitch).unsqueeze(0))
    return soft_units, result


def get_hparams():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='./input',
                        help="Input wav folder")
    parser.add_argument('-m', '--model', type=str, default='./logs/Nyarumul',
                        help='Input model folder')
    parser.add_argument('-o', '--output', type=str, default='./output',
                        help='Output wav folder')
    parser.add_argument('-p', '--pitch', type=int, default=4,
                        help='Pitch shift')
    parser.add_argument('-s', '--singer', type=int, default=0,
                        help='Singer used during generation')
    return parser.parse_args()


def main():
    logger = get_logger()
    hps = get_hparams()
    if not os.path.exists(hps.model):
        logger.warning(f'Cannot find model in path {hps.model}, exiting...')
        return
    if not os.path.exists(hps.input):
        os.makedirs(hps.input)
    if not os.path.exists(hps.output):
        os.makedirs(hps.output)
    if not os.path.exists(os.path.join(hps.model, 'config.json')):
        logger.warning(f'Cannot find model config.json under path {hps.model}')
        return
    with open(os.path.exists(os.path.join(hps.model, 'config.json')), mode='r', encoding='utf-8') as f:
        config = json.load(f)
        data = config.data
        hop_size = getattr(data, 'hop_length', 320)
        sr = getattr(data, 'sampling_rate', 32000)
        net = SynthesizerTrn(
            178,
            config.data.filter_length // 2 + 1,
            config.train_segment_size // hop_size,
            n_speakers=hps.data.n_speakers,
            **hps.model
        )
        _ = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model),
            net,
            None
        )
        sid = hps.singer

        soft_units, pitch = get_pitch(os.listdir(hps.input))
        with torch.inference_mode():
            # TODO("使用 batch 批量推理")
            for index, single_soft_units in enumerate(soft_units):
                single_pitch = pitch[index]
                x_length = torch.LongTensor([single_soft_units.size(1)])
                audio = net.infer(single_soft_units, x_length, single_pitch,
                                  sid, 0.3)[0][0, 0].data.float.numpy()
                # TODO("根据文件名保存")
                sf.write(f'{index}.wav', audio, sr)


if __name__ == '__main__':
    main()
