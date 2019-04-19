from typing import Union, List, Tuple, Iterable
from functools import reduce

from tqdm import tqdm

from nnn.layers import *


class Sequential(Layer):
    def __init__(self, layers: List[types.Layer]):
        super(Sequential, self).__init__(layers[0].input_dim, layers[-1].output_dim)

        self.layers: Iterable[types.Layer] = layers
        self.optimizer: Optional[types.Optimizer] = None
        self.loss: Optional[types.Loss] = None
        self.metric: Optional[types.Metric] = None

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
        super(Sequential, self).update(self.optimizer)

        any(map(lambda layer: layer.update(self.optimizer),
                filter(lambda layer: issubclass(type(layer), types.Layer), self.layers)))

        next(self.optimizer)

    def compile(self,
                optimizer: Union[str, types.Optimizer],
                loss: Union[str, types.Loss],
                metrics: List[Union[str, types.Metric]]):

        self.optimizer = optimizer if isinstance(optimizer, types.Optimizer) \
            else getattr(types.optimizers, optimizer)()
        self.loss = loss if isinstance(loss, types.Loss) \
            else getattr(types.losses, loss)()
        self.metric = metrics[0] if isinstance(metrics[0], types.Metric) \
            else getattr(types.metrics, metrics[0])()

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

    @property
    def parameters(self) \
            -> Iterable[np.ndarray]:
        for layer in self.layers:
            yield layer.parameters
