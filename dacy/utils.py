import numpy as np


def softmax(x):
    return np.exp(x) / sum(np.exp(x))
