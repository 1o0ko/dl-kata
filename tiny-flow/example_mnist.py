'''
Training example with MNIST and feed forward MLP
'''
import logging
import mnist
import numpy as np

from tinyflow.data import chainable
from tinyflow.ops import Input, Linear, Sigmoid, CrossEntropyWithLogits

@chainable
def normalize(X):
    ''' Normalizes RGB data '''
    return X / np.max(X)


@chainable
def flatten(X):
    ''' Flattens image data'''
    assert len(X.shape) == 3
    n, h, w = X.shape

    return X.reshape(n, h * w)


def one_hot(X, n=None):
    n = n if n else np.max(X) + 1
    one_hot_encoding = np.zeros([X.shape[-1], n])
    one_hot_encoding[np.arange(X.shape[-1]), X] = 1

    return one_hot_encoding


def get_weights(n_in, n_out):
    '''
    Initializes weights

    TODO: Add smarter intialization
    '''
    W = np.random.randn(n_in, n_out)
    b = np.random.randn(n_out)

    return W, b


def get_model():
    '''
    Builds the graph
    '''
    X, y = Input(name='X'), Input(name='y')

    W1, b1 = Input(name='W1'), Input(name='b1')
    W2, b2 = Input(name='W2'), Input(name='b2')
    W3, b3 = Input(name='W3'), Input(name='b3')

    l1 = Linear(X, W1, b1, name='l1')
    s1 = Sigmoid(l1, name='s1')
    l2 = Linear(s1, W2, b2, name='l2')
    s2 = Sigmoid(l2, name='s2')
    l3 = Linear(s2, W3, b3, name='l3')

    cost = CrossEntropyWithLogits(y, l3, name='loss')

    return cost

def main():
    # create pipeline
    image_pipeline = normalize >> flatten
    label_pipeline = one_hot

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Loading mnist")
    train_images = image_pipeline(mnist.train_images())
    train_labels = label_pipeline(mnist.train_labels())
    logger.info("Train: %s, %s", train_images.shape, train_labels.shape)

    test_images = image_pipeline(mnist.test_images())
    test_labels = label_pipeline(mnist.test_labels())
    logger.info("Test: %s, %s", test_images.shape, test_labels.shape)
    logger.info("Done!")


if __name__ == '__main__':
    main()
