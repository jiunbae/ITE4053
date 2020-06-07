from tensorflow.keras.utils import Sequence
from tensorflow.keras import optimizers as optim

from network import SuperResolutionNetwork
from utils.callback import CustomCallback


class SuperResolutionModel(object):

    def __init__(self, mode: str):
        self.klass = SuperResolutionNetwork(mode=mode)
        self.model = self.klass()

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
                CustomCallback(log_dir=config.log, interval=config.interval,
                               train=train_generator[0], test=[v for v in val_generator]),
            ]
        )

    def save(self, path: str):
        self.model.save(path)
