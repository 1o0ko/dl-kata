'''
This module contains training procedures.
'''
import numpy as np


class Optimizer(object):
    def __init__(self, trainables=None, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.trainables = trainables if trainables else None

    def update(self, global_step):
        raise NotImplemented


class SGD(Optimizer):
    def __init__(self, trainables, learning_rate):
        Optimizer.__init__(self, trainables, learning_rate)

    def update(self):
        for t in self.trainables:
            t.value -= self.learning_rate * t.gradients[t]


class Momentum(Optimizer):
    def __init__(self, trainables, learning_rate, momentum=0.9):
        Optimizer.__init__(self, trainables, learning_rate)
        self.momentum = momentum
        self.v = {}

    def update(self):
        for t in self.trainables:
            if t not in self.v:
                self.v[t] = np.zeros_like(t.value)

            self.v[t] = self.momentum * self.v[t] - \
                self.learning_rate * t.gradients[t]

            t.value += self.v[t]
