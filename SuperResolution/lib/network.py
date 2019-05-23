import tensorflow as tf
from tensorflow.keras import models as KM
from tensorflow.keras import layers as KL


class SuperResolutionNetwork(object):

    def __new__(cls, shape: tuple = (32, 32, 1)) \
            -> KM.Model:

        image = KL.Input(shape=[None, None, 1],
                         name="input_image")

        layer1 = KL.Conv2D(64, (3, 3), padding="SAME", activation='relu',
                           name="layer1")(image)
        layer2 = KL.Conv2D(64, (3, 3), padding="SAME", activation='relu',
                           name="layer2")(layer1)
        layer3 = KL.Conv2D(1, (3, 3), padding="SAME",
                           name="layer3")(layer2)

        return KM.Model([image], [layer3],
                        name='SR')

    @staticmethod
    def metric(y_true: tf.Tensor, y_pred: tf.Tensor) \
            -> tf.Tensor:
        return tf.image.psnr(y_true, y_pred, max_val=255.)

    @classmethod
    def compile(cls, model, optimizer, loss, metric)\
            -> None:

        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=[metric])
