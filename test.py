import os
os.environ[""] = "TRUE"
import hubert.encode

if __name__ == '__main__':
    hubert.encode.get_hubert_soft_encoder(proxy={'http': 'http://localhost:7891'})