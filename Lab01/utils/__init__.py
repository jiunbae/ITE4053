import random

import numpy as np
import tensorflow as tf


def init(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_random_seed(seed)


from utils.data import Dataset
from utils.arguments import arguments

init(arguments.seed)
