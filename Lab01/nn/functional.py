import numpy as np

from core import _Module


class _Activation(_Module):
    pass


class Sigmoid(_Activation):
    def forward(self, X: np.ndarray, grad: bool = True) -> np.ndarray:
        super(Sigmoid, self).forward(X, grad)

        return 1. / (1. + np.exp(-X))

    def backward(self, grad: np.ndarray) -> np.ndarray:
        last = super(Sigmoid, self).backward(grad)

        result = self.forward(last, grad=False)

        return grad * result * (1. - result)


class ReLU(_Activation):
    def forward(self, X: np.ndarray, grad: bool = True) -> np.ndarray:
        super(ReLU, self).forward(X, grad)

        return np.maximum(0, X)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        last = super(ReLU, self).backward(grad)

        grad = grad.copy()
        grad[last <= 0] = 0

        return grad


class _Loss(_Module):
    name = '_Loss'

    def __int__(self, *args):
        super(_Loss, self).__init__()

    def __call__(self, inputs: np.ndarray, targets: np.ndarray) \
            -> np.ndarray:
        return self.forward(inputs, targets)

    def forward(self, inputs: np.ndarray, targets: np.ndarray) \
            -> np.ndarray:
        return np.zeros(0)


class MSELoss(_Loss):
    name = 'mean_squared_error'

    def __int__(self, axis=None):
        super(MSELoss, self).__init__()

        self.axis = axis

    def forward(self, inputs: np.ndarray, targets: np.ndarray) \
            -> np.ndarray:
        super(MSELoss, self).forward(inputs, targets)

        cost = (np.square(targets - inputs)).mean(axis=self.axis)

        return cost


class BCELoss(_Loss):
    name = 'binary_crossentropy'

    def forward(self, inputs: np.ndarray, targets: np.ndarray) \
            -> np.ndarray:
        super(BCELoss, self).forward(inputs, targets)

        size = np.size(inputs, -1)
        cost = -1 / size * (np.dot(targets, np.log(inputs).T) + np.dot(1 - targets, np.log(1 - inputs).T))

        return np.squeeze(cost)


Losses = {
    klass.name: klass for klass in _Loss.__subclasses__()
}
