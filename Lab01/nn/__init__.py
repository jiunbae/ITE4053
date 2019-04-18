import warnings

import numpy as np


class _Module(object):
    def __init__(self, *args):
        self._last_input = None
        self._last_output = None

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

    def update(self, *args):
        pass


from core.optimizers import _Optimizer, Optimizers
from core.functional import _Activation, _Loss, Losses
from core.metrics import _Metric, Metrics
from core.layers import _Layer
