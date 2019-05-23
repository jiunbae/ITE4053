from typing import Tuple, List
from pathlib import Path
from functools import reduce

import numpy as np
import tensorflow.keras as keras
import cv2

from utils.transform import Transform


class Dataset(keras.utils.Sequence):
    def __init__(self, root: str,
                 size: Tuple[int, int],
                 source_transforms: List[Transform] = None,
                 target_transforms: List[Transform] = None,
                 ext: str = 'bmp', batch: int = 32, shuffle: bool = True):
        self.root = Path(root)
        self.ext = ext
        self.batch = batch
        self.shuffle = shuffle
        self.channels = 1
        self.size = size

        self.source_transforms = source_transforms or []
        self.target_transforms = target_transforms or []

        self.images = list(sorted(self.root.glob(f'*.{self.ext}')))
        self.indices = np.arange(len(self.images))

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
            image = cv2.imread(str(self.images[indices[b]]), cv2.IMREAD_GRAYSCALE)

            target = reduce(lambda i, t: t(i), [image] + self.target_transforms)
            source = reduce(lambda i, t: t(i), [target] + self.source_transforms)

            if self.size != target.shape:
                sources = np.empty((self.batch, *target.shape, self.channels), dtype=np.float32)
                targets = np.empty((self.batch, *target.shape, self.channels), dtype=np.float32)

            sources[b] = np.expand_dims(source, axis=-1)
            targets[b] = np.expand_dims(target, axis=-1)

        return sources, targets
