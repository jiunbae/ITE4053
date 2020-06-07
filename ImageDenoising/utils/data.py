from typing import Tuple, List
from pathlib import Path
from itertools import chain
from functools import reduce

import numpy as np
import tensorflow.keras as keras

from utils.transform import Transform


class Dataset(keras.utils.Sequence):
    def __init__(self, train: bool = True,
                 source_transforms: List[Transform] = None,
                 target_transforms: List[Transform] = None,
                 batch: int = 32, shuffle: bool = True):

        self.batch = batch
        self.shuffle = shuffle
        self.channels = 3
        self.is_training = True

        (self.x_train, _), (self.x_test, _) = keras.datasets.cifar10.load_data()
        self.images = self.x_train
        self.size = self.x_train[0].shape[:2]

        self.source_transforms = source_transforms or []
        self.target_transforms = target_transforms or []

        self.indices = np.arange(len(self.x_train))

    def train(self, flag: bool = True):
        self.is_training = flag

    def eval(self):
        self.train(False)

    def on_epoch_end(self) \
            -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) \
            -> int:
        return len(self.images)

    def __getitem__(self, item: int) \
            -> Tuple[np.ndarray, np.ndarray]:

        sources = np.empty((self.batch, *self.size, self.channels), dtype=np.float32)
        targets = np.empty((self.batch, *self.size, self.channels), dtype=np.float32)

        indices = np.roll(self.indices, item)

        for b in range(self.batch):
            image = self.images[indices[b]]

            sources[b] = reduce(lambda i, t: t(i), [image / 255.] + self.source_transforms)
            targets[b] = reduce(lambda i, t: t(i), [image / 255.] + self.target_transforms)

        return sources, targets
