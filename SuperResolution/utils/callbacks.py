from itertools import chain

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, Callback


class CustomCallback(Callback):

    def __init__(self, log_dir: str, interval: int,
                 train: np.ndarray, test: np.ndarray):
        super(CustomCallback, self).__init__()
        self.train = train
        self.test = test
        self.interval = interval

        self.writer = tf.summary.FileWriter(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        def _summary_image(*spt, tag):
            def _squeeze(img, limit: int = 512):
                if img.shape[0] <= limit and img.shape[1] <= limit:
                    return img

                image = np.zeros((limit, limit, 1))

                target_w, target_h, _ = map(int, np.array(img.shape) / (np.max(img.shape) / limit))

                x, y = int((limit - target_w) / 2), int((limit - target_h) / 2)

                image[x:x+target_w, y:y+target_h] = np.expand_dims(cv2.resize(img, (target_h, target_w)), axis=-1)

                return image

            s, p, t = map(_squeeze, spt)
            img = np.hstack((s * 255., p * 255., t * 255.))
            summary = tf.Summary.Image(encoded_image_string=cv2.imencode('.jpg', img)[1].tostring())
            return tf.Summary.Value(tag=tag, image=summary)

        if not (epoch % self.interval):
            self.writer.add_summary(tf.Summary(value=list(chain(
                map(lambda istp: _summary_image(*istp[1],
                                                tag=f'train_{istp[0]}'),
                    enumerate(zip(self.train[0], self.model.predict(self.train), self.train[1]))),
                map(lambda it: _summary_image(it[1][0][0], self.model.predict(it[1])[0], it[1][1][0],
                                              tag=f'test_{it[0]}'),
                    enumerate(self.test)),
                map(lambda kv: tf.Summary.Value(tag=kv[0], simple_value=kv[1]), logs.items())
            ))), epoch)
