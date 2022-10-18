import numpy as np
import torch as t

if __name__ == '__main__':
    tensor = t.arange(100)
    print(tensor.unsqueeze(0))