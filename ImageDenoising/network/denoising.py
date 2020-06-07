import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import models as KM
from tensorflow.keras import layers as KL


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
