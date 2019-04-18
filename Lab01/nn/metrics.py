import numpy as np


class Metric:

    def __call__(self, inputs: np.ndarray, targets: np.ndarray) \
            -> float:
        return 0


class Accuracy(Metric):

    def __call__(self, inputs: np.ndarray, targets: np.ndarray) \
            -> float:
        return (inputs == targets).all(axis=0).mean()
