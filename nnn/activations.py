import numpy as np

from nnn.core import _Module


class Activation(_Module):
    pass


class Sigmoid(Activation):
    name = 'sigmoid'

    def forward(self, X: np.ndarray, grad: bool = True) -> np.ndarray:
        super(Sigmoid, self).forward(X, grad)

        return 1. / (1. + np.exp(-X))

    def backward(self, grad: np.ndarray) -> np.ndarray:
        last = super(Sigmoid, self).backward(grad)

        result = self.forward(last, grad=False)

        return grad * result * (1. - result)


class ReLU(Activation):
    name = 'relu'
    
    def forward(self, X: np.ndarray, grad: bool = True) -> np.ndarray:
        super(ReLU, self).forward(X, grad)

        return np.maximum(0, X)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        last = super(ReLU, self).backward(grad)

        grad = grad.copy()
        grad[last <= 0] = 0

        return grad
