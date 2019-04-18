import numpy as np

from nnn.core import _Module


class Loss(_Module):
    def __int__(self, *args):
        super(Loss, self).__init__()

    def __call__(self, inputs: np.ndarray, targets: np.ndarray) \
            -> np.ndarray:
        return self.forward(inputs, targets)

    def forward(self, inputs: np.ndarray, targets: np.ndarray) \
            -> np.ndarray:
        return np.zeros(0)


class MSELoss(Loss):
    name = 'mean_squared_error'

    def __int__(self, *args):
        super(MSELoss, self).__init__(*args)


    def forward(self, inputs: np.ndarray, targets: np.ndarray) \
            -> np.ndarray:
        super(MSELoss, self).forward(inputs, targets)

        cost = np.mean((inputs - targets) ** 2)

        return cost


class BCELoss(Loss):
    name = 'binary_crossentropy'

    def __init__(self, *args):
        super(BCELoss, self).__init__(*args)

    def forward(self, inputs: np.ndarray, targets: np.ndarray) \
            -> np.ndarray:
        super(BCELoss, self).forward(inputs, targets)

        size = np.size(inputs, -1)
        cost = -1 / size * (np.dot(targets, np.log(inputs).T) + np.dot(1 - targets, np.log(1 - inputs).T))

        return np.squeeze(cost)

