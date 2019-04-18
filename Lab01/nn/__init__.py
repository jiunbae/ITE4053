import warnings

import numpy as np


class _Module(object):
    def __init__(self, *args):

        self._last_input, self._last_output = None, None

    def __call__(self, X: np.ndarray) \
            -> np.ndarray:
        return self.forward(X)

    def forward(self, X: np.ndarray, grad: bool = True) \
            -> np.ndarray:
        if grad:
            if self._last_input is not None:
                warnings.warn("backward is not called after forward.")
            self._last_input = X

        return self._last_input

    def after_forward(self, output: np.ndarray) \
            -> np.ndarray:
        self._last_output = output

        return self._last_output

    def backward(self, Y: np.ndarray) \
            -> np.ndarray:

        if self._last_input is None:
            raise Exception("backward call must after forward.")

        result = self._last_input
        self._last_input = None

        return result

    def after_backward(self, output: np.ndarray) \
            -> np.ndarray:

        return output

    def update(self, *args):
        pass


from nn.activations import Activation
from nn.optimizers import Optimizer
from nn.functional import Loss
from nn.metrics import Metric
from nn.layers import Layer
from nn.modules import *


def caller_wrapper(f):
    return lambda *args, **kwargs: f(*args, **kwargs)


activations = type('Activations', (object,), {
    klass.__name__.lower(): caller_wrapper(klass) for klass in Activation.__subclasses__()
})

optimizers = type('Optimizers', (object,), {
    klass.__name__: caller_wrapper(klass) for klass in Optimizer.__subclasses__()
})

losses = type('Losses', (object,), {
    klass.name.lower(): caller_wrapper(klass) for klass in Loss.__subclasses__()
})

metrics = type('Metrics', (object, ), {
    klass.__name__.lower(): caller_wrapper(klass) for klass in Metric.__subclasses__()
})
