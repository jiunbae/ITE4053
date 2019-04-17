from typing import Union, List, Tuple
from functools import reduce

from tqdm import tqdm

import core as types
from core import _Module
from core.layer import *


class Sequential(_Module):
    def __init__(self, *layers: types._Layer):
        super(Sequential, self).__init__()

        self.layers = layers
        self.optimizer = None
        self.loss = None
        self.metric = None

    def forward(self, X: np.ndarray, grad: bool = True) \
            -> np.ndarray:
        super(Sequential, self).forward(X, grad)

        result = reduce(lambda inputs, layer: layer.forward(inputs, grad), [X, *self.layers])

        return self.after_forward(result)

    def backward(self, Y: np.ndarray) \
            -> np.ndarray:
        super(Sequential, self).backward(Y)

        grad = -(np.divide(Y, self._last_output) - np.divide(1 - Y, 1 - self._last_output))

        result = reduce(lambda g, layer: layer.backward(g), [grad, *reversed(self.layers)])

        return result

    def update(self, lr: float):
        super(Sequential, self).update(lr)

        for layer in filter(lambda l: issubclass(l.__class__, types._Layer), self.layers):
            layer.update(self.optimizer.lr)

    def compile(self,
                optimizer: Union[str, types._Optimizer],
                loss: Union[str, types._Loss],
                metrics: List[Union[str, types._Metric]]):

        self.optimizer = optimizer if isinstance(optimizer, types._Optimizer) \
            else types.Optimizers.get(optimizer)()
        self.loss = loss if isinstance(loss, types._Loss) \
            else types.Losses.get(loss, loss)()
        self.metric = metrics[0] if isinstance(metrics[0], types._Metric) \
            else types.Metrics.get(metrics[0], metrics[0])()

    def fit(self, X: np.ndarray, Y: np.ndarray,
            epochs: int = 1000, verbose: bool = True):

        t = tqdm(total=epochs) if verbose else None

        for _ in range(epochs):
            forward = self.forward(X.T)
            loss = self.loss(forward, Y)
            self.backward(Y)
            self.update(self.optimizer.lr)

            if verbose and t:
                t.set_postfix(loss=f'{loss:.4f}')
                t.update()

    def evaluate(self, X: np.ndarray, Y: np.ndarray,
                 verbose: bool = True) \
            -> Tuple[float, float]:

        forward = self.forward(X.T, grad=False)
        loss = self.loss(forward, Y)

        forward[forward > .5] = 1.
        forward[forward <= .5] = 0.

        return loss, self.metric(forward, Y)

    @property
    def size(self) -> int:
        return sum(map(lambda x: x.size, self.layers))
