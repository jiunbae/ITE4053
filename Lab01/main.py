import argparse
from typing import Tuple, Union

from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

from utils import init
from utils.data import Dataset
import core.nn as nn
import core.functional as F
import core.optimizer as optim


def get_model(**kwargs) \
        -> Tuple[Union[keras.Sequential, nn.Sequential],
                 Union[keras.optimizers.Optimizer, optim._Optimizer]]:
    return {
        'np': nn.Sequential(
            nn.Linear(2, 2),
            F.Sigmoid(),
            nn.Linear(2, 1),
            F.Sigmoid(),
        ),
        'tf': keras.Sequential([
            keras.layers.Flatten(input_shape=(2,)),
            keras.layers.Dense(2, activation=tf.nn.sigmoid),
            keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ]),
    }[kwargs['mode']], {
        'np': optim.Optimizers.get('adam')(lr=kwargs['lr']),
        'tf': keras.optimizers.Adam(lr=kwargs['lr']),
    }[kwargs['mode']]


def main(args: argparse.Namespace):

    total_score = .0

    with tqdm(total=args.repeat) as t:
        for e in range(args.repeat):
            train = Dataset(args.size)
            test = Dataset(args.size)

            model, optimizer = get_model(
                mode=args.mode,
                lr=args.lr,
            )
            model.compile(optimizer=optimizer,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

            model.fit(train.X, train.Y,
                      epochs=args.epoch, verbose=False)
            loss, acc = model.evaluate(test.X, test.Y,
                                       verbose=False)
            total_score += acc

            t.set_postfix(loss=f'{loss:.4f}', score=f'{acc:.2f}%', mean=f'{total_score / (e+1):.2f}%')
            t.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ITE4053 Assignment 1'
                                                 'Neural Network implemented by numpy and TensorFlow')

    parser.add_argument('--mode', type=str, default='np', choices=['np', 'tf'],
                        help='Select mode for assignment')

    parser.add_argument('--epoch', type=int, default=5000,
                        help="Epoch size")
    parser.add_argument('--size', type=int, default=128,
                        help="Dataset size")
    parser.add_argument('--lr', type=float, default=.1,
                        help="Learning rate")
    parser.add_argument('--repeat', type=int, default=10,
                        help="Repeat train, valid")

    parser.add_argument('--seed', type=int, default=42,
                        help="Manual seed")

    arguments = parser.parse_args()

    init(arguments.seed)

    main(arguments)
    # tfmain(arguments)
