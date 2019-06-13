from .cnn import SuperResolutionNetwork as CNN
from .rnn import SuperResolutionNetwork as RNN


def SuperResolutionNetwork(mode: str):
    return {
        'cnn': CNN,
        'rnn': RNN,
    }[mode.lower()]
