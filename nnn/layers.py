from typing import Optional

import numpy as np

import nnn as types
from nnn.core import _Module


class Layer(_Module):
    def __init__(self, output_dim: int, input_dim: int,
                 activation: Optional[types.Activation] = None,
                 *args):
        super(Layer, self).__init__(*args)

        self.output_dim = output_dim
        self.input_dim = input_dim
        self.activation = activation and activation()

        self.dW, self.db = 0, 0

    def __call__(self, X: np.ndarray, grad: bool = True) \
            -> np.ndarray:
        return self.forward(X, grad)

    def after_forward(self, output: np.ndarray) \
            -> np.ndarray:
        output = super(Layer, self).after_forward(output)

        if self.activation is not None:
            output = self.activation.forward(output)

        return output

    def backward(self, grad: np.ndarray) \
            -> np.ndarray:

        if self.activation is not None:
            grad = self.activation.backward(grad)

        return grad

    def update(self, optimizer: types.Optimizer):
        super(Layer, self).update()

        # clear last input and output for next step
        _ = self.last_input, self.last_output

    @property
    def last_input(self) \
            -> np.ndarray:

        result = self._last_input
        self._last_input = None
        return result

    @property
    def last_output(self)\
            -> np.ndarray:

        result = self._last_output
        self._last_output = None
        return result

    @property
    def parameters(self) \
            -> np.ndarray:

        return np.empty(0)


class Dense(Layer):
    def __init__(self, output_dim: int, input_dim: int,
                 activation: Optional[types.Activation] = None,
                 *args):
        super(Dense, self).__init__(output_dim, input_dim, activation, *args)

        self.params = np.random.randn(self.output_dim, self.input_dim) * .1
        self.bias = np.random.randn(self.output_dim, 1) * .1

    def forward(self, X: np.ndarray, grad: bool = True) \
            -> np.ndarray:
        super(Dense, self).forward(X, grad)

        result = np.dot(self.params, X) + self.bias

        return self.after_forward(result)

    def backward(self, grad: np.ndarray) \
            -> np.ndarray:
        grad = super(Dense, self).backward(grad)

        size = np.size(grad, -1)

        self.dW = np.dot(grad, self._last_input.T) / size
        self.db = np.sum(grad, axis=1, keepdims=True) / size

        result = np.dot(self.params.T, grad)

        return result

    def update(self, optimizer: types.Optimizer):
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
