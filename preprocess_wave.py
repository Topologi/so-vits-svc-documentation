import argparse
import logging
import os
import re

import librosa
import numpy as np
import torch.utils.data
import torchaudio
from sounds_feature import FeatureInput

import utils
from hubert import encode


# Thanks to IceKyrin
# https://github.com/IceKyrin/sovits_guide
def resize2d(x, target_len):
    source = np.array(x)
    source[source < 0.001] = np.nan
    target = np.interp(np.arange(0, len(source) * target_len, len(source)) / target_len, np.arange(0, len(source)),
                       source)
    res = np.nan_to_num(target)
    return res


def get_logger():
    fo = '%(asctime)s %(message)s'
    logging.basicConfig(format=fo)
    logger_ = logging.getLogger('SoundPreprocessor')
    logger_.setLevel(logging.INFO)
    return logger_


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
    parser.add_argument('-p', '--proxy', type=str, default="",
                        help="Proxy server for downloading HuBERT model, example: http(s)://localhost:7891")
    parser.add_argument('-v', '--valid', type=str, default="./datasets/valid",
                        help="Where to save validation datasets")
    parser.add_argument('-pa', '--partition', type=bool, default=False)
    return parser.parse_args()


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, sound_folder, output_hubert_folder):
        super().__init__()
        self.sound_folder = sound_folder
        self.output_hubert_folder = output_hubert_folder
        self.filename = []
        for filename in os.listdir(sound_folder):
            if filename.endswith('.wav'):
                self.filename.append(filename)

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, index):
        audio, sr = torchaudio.load(os.path.join(self.sound_folder, self.filename[index]))
        if sr != 16000:
            audio = librosa.resample(audio.cpu().numpy(), orig_sr=sr, target_sr=16000)
        return torch.tensor(audio).unsqueeze(0), self.filename[index], self.sound_folder, self.output_hubert_folder


def load_hubert_audio(origin_sounds_folder, output_hubert_folder):
    """
        FIXME: 可以修改实现为 DataLoader，但是要做 Padding，先暂时用单线程搞定
    """
    return AudioDataset(origin_sounds_folder, output_hubert_folder)


def process_wav(wav_path, out_f0, out_speech_units, hubert_net):
    if not os.path.exists(wav_path):
        os.makedirs(wav_path)
    if not os.path.exists(out_f0):
        os.makedirs(out_f0)
    if not os.path.exists(out_speech_units):
        os.makedirs(out_speech_units)

    # define model and load checkpoint
    hps = utils.get_hparams_from_file(args.config)
    # TODO: 增加针对 44100 Hz 的处理
    feature_input = FeatureInput(hps.data.sampling_rate, hps.data.hop_length)
    with open(args.description, "w", encoding="utf-8") as vits_train_data_desc:
        for speaker_id in os.listdir(wav_path):
            if os.path.isfile(os.path.join(wav_path, speaker_id)):
                continue
            if not re.match('[1-8]', speaker_id):
                logger.warning(f'Could not handle speaker with id: {speaker_id}, expected number from 1 to 8')
                continue
            if os.path.isdir(os.path.join(wav_path, speaker_id)):
                if not os.path.exists(os.path.join(out_f0, speaker_id)):
                    os.mkdir(os.path.join(out_f0, speaker_id))
                if not os.path.exists(os.path.join(out_speech_units, speaker_id)):
                    os.mkdir(os.path.join(out_speech_units, speaker_id))
                # 开始执行并行化音频处理

                # 执行 Hubert 处理 #
                count = 0
                audios_dataloader = load_hubert_audio(os.path.join(wav_path, speaker_id), out_speech_units)
                datasets = load_hubert_audio(os.path.join(wav_path, speaker_id),
                                             os.path.join(out_speech_units, speaker_id))

                for data, filename, sound_folder, hubert_folder in datasets:
                    if torch.cuda.is_available():
                        torch.save(hubert_net.units(data.cuda()).squeeze(),
                                   os.path.join(hubert_folder, f'{filename[:-4]}.npy'))
                    else:
                        torch.save(hubert_net.units(data.cpu()).squeeze(),
                                   os.path.join(hubert_folder, f'{filename[:-4]}.npy'))
                    count += 1
                    if count % 10 == 0:
                        logger.info(f'Hubert handled total {count} files')
                del count
                # Hubert 处理结束 #

                for file in os.listdir(os.path.join(wav_path, speaker_id)):
                    if file.endswith(".wav"):
                        # 消除文件后缀名
                        file = file[:-4]
                        audio_path = os.path.join(wav_path, speaker_id, f'{file}.wav')
                        soft = torch.load(os.path.join(out_speech_units, speaker_id, f'{file}.npy'))
                        feature_pit = feature_input.compute_f0(audio_path)
                        # 标准化 f0 尺寸，与 HuBERT 输出对应
                        feature_pit = resize2d(feature_pit, soft.shape[0])
                        coarse_pit = feature_input.coarse_f0(feature_pit)

                        np.save(
                            os.path.join(out_f0, speaker_id, f'{file}_pitch.npy'),
                            coarse_pit,
                            allow_pickle=False,
                        )

                        np.save(
                            os.path.join(wav_path, speaker_id, f'{file}_nsff0.npy'),
                            feature_pit,
                            allow_pickle=False,
                        )

                        # HuBERT code
                        path_label = os.path.join(out_speech_units, speaker_id, f'{file}.npy')
                        path_pitch = os.path.join(out_f0, speaker_id, f'{file}_pitch.npy')
                        path_nsff0 = os.path.join(wav_path, speaker_id, f'{file}_nsff0.npy')
                        windows_slash = '\\'
                        linux_slash = '/'
                        print(
                            f"{audio_path.replace(windows_slash, linux_slash)}|"
                            f"{speaker_id.replace(windows_slash, linux_slash)}|"
                            f"{path_label.replace(windows_slash, linux_slash)}|"
                            f"{path_pitch.replace(windows_slash, linux_slash)}|"
                            f"{path_nsff0.replace(windows_slash, linux_slash)}",
                            file=vits_train_data_desc,
                        )


if __name__ == "__main__":
    numba_logger = logging.getLogger("numba")
    numba_logger.setLevel(logging.WARNING)
    logger = get_logger()
    logger.info('Initializing according to parameters')
    args = get_hparams()
    wavPath = args.sounds
    outF0 = args.f0
    outSpeechUnits = args.units
    validPath = args.valid
    logger.info(f'Input sound path: {wavPath}')
    logger.info(f'Output F0 path: {outF0}')
    logger.info(f'Output HuBERT speech units path: {outSpeechUnits}')
    logger.info(f'Validation datasets path: {validPath}')
    # 开始尝试启动 HuBERT
    proxy = args.proxy
    if len(proxy) == 0:
        proxy_obj = None
    else:
        proxy_obj = {
            'http': proxy,
            'https': proxy
        }
    hubert_net = encode.get_hubert_soft_encoder(proxy_obj)
    if torch.cuda.is_available():
        hubert_net.cuda()
    else:
        hubert_net.cpu()
    # HuBERT 加载完毕
    # 将部分训练集划分到验证集
    if args.partition:
        for spk_id in os.listdir(wavPath):
            if os.path.isfile(os.path.join(wavPath, spk_id)):
                continue
            speaker_wav_path = os.path.join(wavPath, spk_id)
            filenames = list(iter(os.listdir(speaker_wav_path)))
            for item in filenames:
                if not item.endswith('.wav'):
                    filenames.remove(item)
            rand_sel_files_index = torch.randperm(len(filenames))
            rand_sel_files_index = rand_sel_files_index[:int(len(filenames) / 10)]
            if not os.path.exists(os.path.join(validPath, f'./sounds{spk_id}')):
                os.makedirs(os.path.join(validPath, f'./sounds/{spk_id}'))
            for i in rand_sel_files_index:
                ind = i.item()
                filename = filenames[ind]
                os.replace(os.path.join(wavPath, f'./{spk_id}/{filename}'),
                           os.path.join(validPath, f'./sounds/{spk_id}/{filename}'))
    process_wav(wavPath, outF0, outSpeechUnits, hubert_net)
    process_wav(os.path.join(validPath, './sounds'),
                os.path.join(validPath, './f0'),
                os.path.join(validPath, './speech_units'),
                hubert_net)

