import argparse
import logging
import os.path

import numpy as np
from pathlib import Path
from tqdm import tqdm
import requests

import torch
import torchaudio
from torchaudio.functional import resample
from .hubert.model import URLS
from .hubert.model import hubert_soft


def encode_dataset(args):
    print(f"Loading hubert checkpoint")
    hubert = torch.hub.load("bshall/hubert:main", f"hubert_{args.model}").cuda()

    print(f"Encoding dataset at {args.in_dir}")
    for in_path in tqdm(list(args.in_dir.rglob(f"*{args.extension}"))):
        wav, sr = torchaudio.load(in_path)
        wav = resample(wav, sr, 16000)
        wav = wav.unsqueeze(0).cuda()

        with torch.inference_mode():
            units = hubert.units(wav)

        out_path = args.out_dir / in_path.relative_to(args.in_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path.with_suffix(".npy"), units.squeeze().cpu().numpy())


def get_logger():
    fo = '%(asctime)s %(message)s'
    logging.basicConfig(format=fo)
    lo = logging.getLogger('HuBERT')
    lo.setLevel(logging.DEBUG)
    return lo


def check_and_download_model(logger, proxy=None):
    logger.info('Checking for HuBERT content encoder...')
    if not os.path.exists('./hubert/model/hubert-soft.pt'):
        logger.info(f'Not found, downloading{f" with proxy {proxy}" if proxy is not None else ""}')
        with requests.get(URLS['hubert-soft'], proxies=proxy) as r:
            r.raise_for_status()
            folder = './hubert/model'
            if not os.path.exists(folder):
                os.mkdir(folder)
            with open('./hubert/model/hubert-soft.pt', 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    else:
        logger.info('HuBERT content encoder found')


def get_hubert_soft_encoder(proxy=None):
    logger = get_logger()
    check_and_download_model(logger, proxy)
    logger.info('Loading HuBERT model...')
    # TODO: here
    return hubert_soft(True, True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Encode an audio dataset.")
    parser.add_argument(
        "model",
        help="available models (HuBERT-Soft or HuBERT-Discrete)",
        choices=["soft", "discrete"],
    )
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to the dataset directory.",
        type=Path,
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to the output directory.",
        type=Path,
    )
    parser.add_argument(
        "--extension",
        help="extension of the audio files (defaults to .flac).",
        default=".flac",
        type=str,
    )
    args = parser.parse_args()
    encode_dataset(args)
