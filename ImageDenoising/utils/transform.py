from typing import Tuple

import cv2
import numpy as np


class Transform:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, image: np.ndarray):
        pass


class GaussianNoise(Transform):
    def __init__(self, size: Tuple[int, int] = None, mean: float = .0, std: float = .1):
        super(GaussianNoise, self).__init__()
        self.size = size
        self.mean = mean
        self.std = std

    def __call__(self, image: np.ndarray):
        super(GaussianNoise, self).__call__(image)

        image += np.random.normal(self.mean, self.std, self.size)

        return image


class Crop(Transform):
    def __init__(self, size: Tuple[int, int] = None, pos: Tuple[int, int] = None):
        super(Crop, self).__init__()
        self.size = size
        self.pos = pos

    def __call__(self, image: np.ndarray):
        super(Crop, self).__call__(image)

        w, h = self.size or (
            np.random.randint(int(np.size(image, 0) / 2)),
            np.random.randint(int(np.size(image, 1) / 2)),
        )

        x, y = self.pos or (
            np.random.randint(np.size(image, 0) - w),
            np.random.randint(np.size(image, 1) - h),
        )

        return image[x:x + w, y:y + h]


class Resize(Transform):
    def __init__(self, size: Tuple[int, int] = (0, 0), scale: float = 1.,
                 metric=cv2.INTER_CUBIC):
        super(Resize, self).__init__()
        self.size = size
        self.scale = scale
        self.metric = metric

    def __call__(self, image: np.ndarray):

        scale = self.scale

        if self.size == (0, 0) and self.scale == 1.:
            scale = (np.random.rand(1) * .5 + .5)[0]

        return cv2.resize(image, self.size, fx=scale, fy=scale,
                          interpolation=self.metric)
