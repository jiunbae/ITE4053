from typing import Union, List, Tuple, Iterable, Optional
from functools import reduce

from tqdm import tqdm

from core import _Module
from core.layers import *


class Sequential(_Module):
    def __init__(self, layers: List[types._Layer]):
        super(Sequential, self).__init__()

        self.layers: Iterable[types._Layer] = layers
        self.optimizer: Optional[types._Optimizer] = None
        self.loss: Optional[types._Loss] = None
        self.metric: Optional[types._Metric] = None

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

    def update(self):
        super(Sequential, self).update()

        for layer in filter(lambda l: issubclass(l.__class__, types._Layer), self.layers):
            layer.update(self.optimizer)

    def compile(self,
                optimizer: Optional[types._Optimizer],
                loss: Optional[types._Loss],
                metrics: List[Optional[types._Metric]]):

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
            self.update()

            if verbose and t:
                t.set_postfix(loss=f'{loss:.4f}')
                t.update()

    def evaluate(self, X: np.ndarray, Y: np.ndarray,
                 verbose: bool = True) \
            -> Tuple[Union[float, np.ndarray], float]:

        forward = self.forward(X.T, grad=False)
        loss = self.loss(forward, Y)

        forward[forward > .5] = 1.
        forward[forward <= .5] = 0.

        return loss, self.metric(forward, Y)

    @property
    def size(self) \
            -> int:
        return sum(map(lambda x: x.size, self.layers))
