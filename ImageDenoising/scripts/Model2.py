from typing import Tuple, List
import argparse
import random
from pathlib import Path
from itertools import chain
from functools import reduce

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import Sequence
from tensorflow.keras import optimizers as optim
from tensorflow.keras import backend as K
from tensorflow.keras import models as KM
from tensorflow.keras import layers as KL


def init(seed: int):
    random.seed(seed)
    np.random.seed(seed)

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


class Eval:
    def __init__(self, filename: str):
        self.image = np.expand_dims(cv2.imread(filename) / 255., axis=0)
    
    def set_result(self, image: np.ndarray):
        self.image = image
        return self
    
    def to_png(self, filename: str):
        *path, ext = filename.split('.')
        filename = 'Model2.png'
        cv2.imwrite(filename, self.image)

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


class DenoisingNetwork(object):

    def __new__(cls, mode: str) \
            -> KM.Model:
        assert mode in ['base', 'skip', 'bn']

        inputs = KL.Input(shape=[None, None, 3],
                          name="input_image")
        x = inputs
        x = KL.Conv2D(64, (3, 3), padding="SAME",
                      kernel_initializer='random_uniform',
                      bias_initializer='zeros',
                      name="layer1")(x)
        if mode == 'bn':
            x = KL.BatchNormalization()(x)
        x = KL.ReLU()(x)
        x = KL.Conv2D(64, (3, 3), padding="SAME",
                      kernel_initializer='random_uniform',
                      bias_initializer='zeros',
                      name="layer2")(x)
        if mode == 'bn':
            x = KL.BatchNormalization()(x)
        x = KL.ReLU()(x)
        x = KL.Conv2D(64, (3, 3), padding="SAME",
                      kernel_initializer='random_uniform',
                      bias_initializer='zeros',
                      name="layer3")(x)
        if mode == 'bn':
            x = KL.BatchNormalization()(x)
        x = KL.ReLU()(x)
        x = KL.Conv2D(64, (3, 3), padding="SAME",
                      kernel_initializer='random_uniform',
                      bias_initializer='zeros',
                      name="layer4")(x)
        if mode == 'bn':
            x = KL.BatchNormalization()(x)
        x = KL.ReLU()(x)
        x = KL.Conv2D(3, (3, 3), padding="SAME",
                      kernel_initializer='random_uniform',
                      bias_initializer='zeros',
                      name="layer5")(x)
        if mode == 'bn':
            x = KL.BatchNormalization()(x)
        x = KL.ReLU()(x)

        if mode == 'skip' or mode == 'bn':
            x = KL.average([x, inputs])

        return KM.Model(inputs=inputs, outputs=x,
                        name='denoising')

    @staticmethod
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) \
            -> tf.Tensor:
        return K.mean(K.square(y_pred - y_true))

    @classmethod
    def metric(cls, y_true: tf.Tensor, y_pred: tf.Tensor) \
            -> tf.Tensor:
        return tf.image.psnr(y_true, y_pred, max_val=1.)

    @classmethod
    def compile(cls, model, optimizer, loss, metric)\
            -> None:

        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=[metric])


class DenoisingModel(object):

    def __init__(self, mode: str):
        self.klass = DenoisingNetwork
        self.model = self.klass(mode)

    def train(self,
              train_generator: Sequence,
              val_generator: Sequence,
              config: object, epochs: int) \
            -> None:

        optimizer = optim.Adam(lr=config.lr,
                               decay=config.lr_decay)

        self.klass.compile(self.model,
                           optimizer=optimizer,
                           loss=self.klass.loss,
                           metric=self.klass.metric)

        self.model.fit_generator(
            train_generator,
            epochs=epochs,
            steps_per_epoch=len(train_generator),
            validation_data=val_generator,
            validation_steps=100,
            workers=4,
            use_multiprocessing=True,
            callbacks=[
                # TensorBoard(log_dir=config.log, write_graph=True, write_images=True),
                # CustomCallback(log_dir=config.log, interval=config.interval,
                #                train=train_generator[0], test=[v for v in val_generator]),
            ]
        )

    def predict(self, inputs):
        result, *_ = self.model.predict(inputs)
        return result

    def save(self, path: str):
        self.model.save(path)


def main(args: argparse.Namespace):
    train_generator = Dataset(
        batch=args.batch,
        target_transforms=[
        ], source_transforms=[
            GaussianNoise(),
        ]
    )
    val_generator = Dataset(
        train=False,
        batch=1,
        target_transforms=[
        ], source_transforms=[
            GaussianNoise(),
        ]
    )

    model = DenoisingModel(mode=args.mode)
    model.train(train_generator=train_generator,
                val_generator=val_generator,
                epochs=args.epoch, config=args)

    model.save('model.hdf5')

    if args.test:
        eval_dataset = Eval(args.test)
        result = model.predict(eval_dataset.image)
        eval_dataset.set_result(result * 255.).to_png(args.test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Image Denoising')
    parser.add_argument("--mode", default='skip', choices=['base', 'skip', 'bn'],
                        help="Select mode for training model")

    parser.add_argument("--epoch", type=int, default=100, required=False,
                        help="Epoch for training")
    parser.add_argument("--interval", type=int, default=1, required=False)
    parser.add_argument("--batch", type=int, default=32, required=False,
                        help="Mini-batch for training")

    parser.add_argument("--lr", type=float, default=.001, required=False)
    parser.add_argument("--lr-decay", type=float, default=.0, required=False)

    parser.add_argument("--test", type=str, default='noisy.png', required=False,
                        help="Test filename")
    parser.add_argument("--log", type=str, default='./logs', required=False,
                        help="Logging directory")
    parser.add_argument("--seed", type=int, default=42, required=False,
                        help="The answer to life the universe and everything")

    arguments = parser.parse_args()

    init(arguments.seed)

    main(arguments)
