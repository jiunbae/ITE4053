import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras import optimizers as optim
from tensorflow.keras.callbacks import TensorBoard

from lib.network import SuperResolutionNetwork
from utils.callbacks import CustomCallback


class SuperResolutionModel(object):

    def __init__(self, shape: tuple = (32, 32, 1)):
        self.shape = shape

        self.model = SuperResolutionNetwork(shape=shape)

    def train(self,
              train_generator: Sequence,
              val_generator: Sequence,
              config: object, epochs: int) \
            -> None:

        optimizer = optim.Adam(lr=config.lr,
                               decay=config.lr_decay)

        SuperResolutionNetwork.compile(self.model,
                                       optimizer=optimizer,
                                       loss='mean_squared_error',
                                       metric=SuperResolutionNetwork.metric)

        self.model.fit_generator(
            train_generator,
            epochs=epochs,
            steps_per_epoch=1000,
            validation_data=val_generator,
            validation_steps=100,
            workers=4,
            use_multiprocessing=True,
            callbacks=[
                # TensorBoard(log_dir=config.log, write_graph=True, write_images=True),
                CustomCallback(log_dir=config.log, interval=config.interval,
                               train=train_generator[0], test=[v for v in val_generator]),
            ]
        )

    def save(self, path: str):
        self.model.save(path)