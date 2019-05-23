from typing import Tuple

import cv2
import numpy as np


class Transform:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, image: np.ndarray):
        pass


class Crop(Transform):
    def __init__(self, size: Tuple[int, int], pos: Tuple[int, int] = None):
        super(Crop, self).__init__()
        self.size = size
        self.pos = pos

    def __call__(self, image: np.ndarray):
        super(Crop, self).__call__(image)

        x, y = self.pos or (
            np.random.randint(np.size(image, 0) - self.size[0]),
            np.random.randint(np.size(image, 1) - self.size[1]),
        )

        return image[x:x + self.size[0], y:y + self.size[1]]


class Resize(Transform):
    def __init__(self, size: Tuple[int, int] = (0, 0), scale: float = 1.,
                 metric=cv2.INTER_CUBIC):
        super(Resize, self).__init__()
        self.size = size
        self.scale = scale
        self.metric = metric

    def __call__(self, image: np.ndarray):
        return cv2.resize(image, self.size, fx=self.scale, fy=self.scale,
                          interpolation=self.metric)
