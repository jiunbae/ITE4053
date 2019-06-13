import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import models as KM
from tensorflow.keras import layers as KL


class SuperResolutionNetwork(object):

    def __new__(cls) \
            -> KM.Model:

        image = KL.Input(shape=[None, None, 1],
                         name="input_image")

        layer1 = KL.Conv2D(64, (3, 3), padding="SAME", activation='relu',
                           kernel_initializer='random_uniform',
                           bias_initializer='zeros',
                           name="layer1")(image)
        layer2 = KL.Conv2D(64, (3, 3), padding="SAME", activation='relu',
                           kernel_initializer='random_uniform',
                           bias_initializer='zeros',
                           name="layer2")(layer1)
        layer3 = KL.Conv2D(1, (3, 3), padding="SAME",
                           kernel_initializer='random_uniform',
                           bias_initializer='zeros',
                           name="layer3")(layer2)

        return KM.Model([image], [layer3],
                        name='SRN')

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
