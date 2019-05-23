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
        if epoch % self.interval:
            preds = self.model.predict(self.test)

            for index, (source, target, pred) in enumerate(zip(self.test[0], self.test[1], preds)):
                image = np.hstack((source, pred, target))
                summary = tf.Summary.Image(encoded_image_string=cv2.imencode('.jpg', image)[1].tostring())
                summary = tf.Summary(value=[tf.Summary.Value(tag=f'train_{index}', image=summary)])
                self.writer.add_summary(summary, epoch)
