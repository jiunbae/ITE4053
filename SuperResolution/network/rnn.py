from typing import Tuple

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import models as KM
from tensorflow.keras import layers as KL


class RNN(KL.Layer):
    def __init__(self, units, shape: Tuple[int],
                 *args, **kwargs):
        self.units = units
        self.state_size = units
        self.shape = shape
        self.latest = None
        super(RNN, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.input_kernel = self.add_weight(shape=(self.units, self.units), initializer='uniform',
                                            name='input_kernel')

        self.hidden_kernel = self.add_weight(shape=(self.units, self.units), initializer='uniform',
                                             name='hidden_kernel')

        self.output_kernel = self.add_weight(shape=(self.units, self.units), initializer='uniform',
                                             name='output_kernel')

        self.hidden_bias = self.add_weight(shape=(self.units,), initializer='zeros', name='hidden_bias')
        self.output_bias = self.add_weight(shape=(self.units,), initializer='zeros', name='output_bias')
        self.built = True

    def get_initial_state(self, inputs, batch_size, dtype):
        return [tf.zeros((batch_size, self.units, self.units, self.units), dtype)]

    def call(self, inputs, states):
        if len(states[0].shape) == 2:
            prev_hidden = tf.reshape(tf.tile(states[0], [1, 3]), [-1, *inputs.shape[1:-1], states[0].shape[-1]])
        else:
            prev_hidden = states[0]

        if self.latest is None:
            self.latest = inputs

        inputs = tf.concat([inputs, self.latest], axis=-1)

        inputs = KL.Conv2D(32, (3, 3), padding="SAME",
                           name="layer1")(inputs)

        h = K.dot(inputs, self.input_kernel) + K.dot(prev_hidden, self.hidden_kernel) + self.hidden_bias

        output = K.relu(K.dot(h, self.output_kernel) + self.output_bias)

        output = KL.Conv2D(1, (3, 3), padding="SAME",
                           name="layer2")(output)

        self.latest = output

        return output, [h]

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units


class SuperResolutionNetwork(object):

    def __new__(cls, shape: Tuple[int] = (32, 32),
                units: int = 32, repeat: int = 3) \
            -> KM.Model:
        def _preprocess(image):
            image = tf.reshape(tf.tile(tf.squeeze(inputs, -1), [1, repeat, repeat]), [-1, repeat, *shape])
            image = tf.expand_dims(image, -1)
            return image

        def _postprocess(output):
            return output[repeat-1::repeat]

        inputs = KL.Input(shape=[None, None, 1],
                         name="input_image")

        image = KL.Lambda(_preprocess)(inputs)

        output = KL.RNN(RNN(units=units, shape=shape), return_sequences=False,
                        name='rnn')(image)

        output = KL.Lambda(_postprocess)(output)

        return KM.Model([inputs], [output],
                        name='SRN')

    @staticmethod
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) \
            -> tf.Tensor:
        return K.mean(K.square(y_pred - y_true))

    @staticmethod
    def metric(y_true: tf.Tensor, y_pred: tf.Tensor) \
            -> tf.Tensor:
        return tf.image.psnr(y_true, y_pred, max_val=1.)

    @classmethod
    def compile(cls, model, optimizer, loss, metric)\
            -> None:

        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=[metric])
