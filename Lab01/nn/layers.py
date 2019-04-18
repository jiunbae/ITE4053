import numpy as np

import core as types
from core import _Module


class _Layer(_Module):
    def __init__(self, in_dim: int, out_dim: int = 1, *args):
        super(_Layer, self).__init__(*args)

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.dW, self.db = 0, 0

    def __call__(self, X: np.ndarray, grad: bool = True) \
            -> np.ndarray:
        return self.forward(X, grad)

    def update(self, optimizer: types._Optimizer):
        super(_Layer, self).update()

    @property
    def parameters(self) \
            -> np.ndarray:
        return np.empty(0)


class Dense(_Layer):
    def __init__(self, *args):
        super(Dense, self).__init__(*args)

        self.params = np.random.randn(self.out_dim, self.in_dim) * .1
        self.bias = np.random.randn(self.out_dim, 1) * .1

    def forward(self, X: np.ndarray, grad: bool = True) \
            -> np.ndarray:
        super(Dense, self).forward(X, grad)

        result = np.dot(self.params, X) + self.bias

        return self.after_forward(result)

    def backward(self, grad: np.ndarray) \
            -> np.ndarray:
        last = super(Dense, self).backward(grad)

        size = np.size(grad, -1)

        self.dW = np.dot(grad, last.T) / size
        self.db = np.sum(grad, axis=1, keepdims=True) / size

        result = np.dot(self.params.T, grad)

        return result

    def update(self, optimizer: types._Optimizer):
        super(Dense, self).update(optimizer)

        self.parameters = optimizer.get_update(self.parameters, np.hstack([self.dW, self.db]))

        self.dW, self.db = 0, 0

    @property
    def parameters(self) \
            -> np.ndarray:
        return np.hstack([
            self.params,
            self.bias,
        ])

    @parameters.setter
    def parameters(self, parameters: np.ndarray):
        self.params = parameters[:, :-1].reshape(self.params.shape)
        self.bias = parameters[:, -1].reshape(self.bias.shape)

    @property
    def size(self) \
            -> int:
        return self.parameters.size
