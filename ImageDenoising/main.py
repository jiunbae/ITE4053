import argparse
from pathlib import Path

from lib.model import DenoisingModel

import utils


def main(args: argparse.Namespace):
    train_generator = utils.data.Dataset(
        batch=args.batch,
        target_transforms=[
        ], source_transforms=[
            utils.transform.GaussianNoise(),
        ]
    )
    val_generator = utils.data.Dataset(
        train=False,
        batch=1,
        target_transforms=[
        ], source_transforms=[
            utils.transform.GaussianNoise(),
        ]
    )

    model = DenoisingModel(mode=args.mode)
    model.train(train_generator=train_generator,
                val_generator=val_generator,
                epochs=args.epoch, config=args)

    model.save('model.hdf5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Multi Object Tracking')
    parser.add_argument("command", metavar="<command>",
                        choices=['train', 'eval'],
                        help="'train' or 'eval'")
    parser.add_argument("--mode", default='base', choices=['base', 'skip', 'bn'],
                        help="Select mode for training model")

    parser.add_argument("--epoch", type=int, default=100, required=False,
                        help="Epoch for training")
    parser.add_argument("--interval", type=int, default=100, required=False)
    parser.add_argument("--batch", type=int, default=32, required=False,
                        help="Mini-batch for training")

    parser.add_argument("--lr", type=float, default=.001, required=False)
    parser.add_argument("--lr-decay", type=float, default=.0, required=False)

    parser.add_argument("--log", type=str, default='./logs', required=False,
                        help="Logging directory")
    parser.add_argument("--seed", type=int, default=42, required=False,
                        help="The answer to life the universe and everything")

    arguments = parser.parse_args()

    utils.init(arguments.seed)

    main(arguments)
