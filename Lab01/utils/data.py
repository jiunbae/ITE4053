from typing import Tuple, Callable

import numpy as np


class Dataset(object):
    def __init__(self, size: int, shape: Tuple = (2, )):
        self.size = int(size)
        self.shape = shape

        self.X = np.random.rand(self.size, *shape)
        self.Y = np.zeros(self.size)

        self.Y[self.X[:, 0] * self.X[:, 0] > self.X[:, 1]] = 1
