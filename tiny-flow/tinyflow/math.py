import numpy as np


def sigmoid(x):
    ''' Calculates sigmoid function '''
    return 1./(1.+np.exp(-x))

def softmax(x):
    ''' Calculates softmax function '''
    exp_x = np.exp(x)
    probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return probs

