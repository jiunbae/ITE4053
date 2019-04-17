import random

import numpy as np
import tensorflow as tf


def init(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_random_seed(seed)
