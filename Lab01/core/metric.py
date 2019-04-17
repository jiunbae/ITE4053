import numpy as np


class _Metric:
    name = '_Metric'
    pass


class Accuracy(_Metric):
    name = 'accuracy'

    def __call__(self, inputs: np.ndarray, targets: np.ndarray) \
            -> float:
        return (inputs == targets).all(axis=0).mean()


Metrics = {
    klass.name: klass for klass in _Metric.__subclasses__()
}
