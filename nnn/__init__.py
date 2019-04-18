from typing import Callable

from nnn.activations import Activation
from nnn.optimizers import Optimizer
from nnn.functional import Loss
from nnn.metrics import Metric
from nnn.layers import Layer
from nnn.modules import *


def collect(name: str, klass: type) \
        -> object:

    def caller_wrapper(f: Callable) \
            -> object:
        return lambda *args, **kwargs: f(*args, **kwargs)

    return type(name, (object, ), {
        getattr(k, 'name', k.__name__): caller_wrapper(k)
        for k in klass.__subclasses__()
    })
    pass


activations = collect('Activations', Activation)
optimizers = collect('Optimizers', Optimizer)
losses = collect('Losses', Loss)
metrics = collect('Metrics', Metric)
