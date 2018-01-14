'''
Training example with MNIST and feed forward MLP
'''
import logging
import mnist
import numpy as np

from tinyflow.data import chainable


@chainable
def normalize(X):
    ''' Normalizes RGB data '''
    return X / np.max(X)


@chainable
def flatten(X):
    ''' Flattens image data'''
    assert len(X.shape) == 3
    n, h, w = X.shape

    return X.reshape(n, h*w)


def main():
    # create pipeline
    pipeline = normalize >> flatten

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Loading mnist")
    train_images = pipeline(mnist.train_images())
    train_labels = mnist.train_labels()
    logger.info("Train: %s, %s", train_images.shape, train_labels.shape)
    logger.info(train_images[0], train_labels[0])

    test_images = pipeline(mnist.test_images())
    test_labels = mnist.test_labels()
    logger.info("Test: %s, %s", test_images.shape, test_labels.shape)
    logger.info("Done!")


if __name__ == '__main__':
    main()
