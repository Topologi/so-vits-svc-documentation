import logging
import os.path

import requests

from .hubert.model import URLS
from .hubert.model import hubert_soft


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
    return hubert_soft(True, True, False)
