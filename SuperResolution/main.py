import argparse
from pathlib import Path

from lib.model import SuperResolutionModel
import utils


def main(args: argparse.Namespace):
    data_root = Path(args.dataset)

    source_size = (args.source_size, args.source_size)
    target_size = (args.target_size, args.target_size)

    train_generator = utils.data.Dataset([
        str(data_root.joinpath('91')),
        str(data_root.joinpath('291')),
    ],
        batch=128,
        size=target_size,
        target_transforms=[
            utils.transform.Crop(target_size),
        ], source_transforms=[
            utils.transform.Resize(source_size),
            utils.transform.Resize(target_size),
        ]
    )

    val_generator = utils.data.Dataset(
        str(data_root.joinpath('Set5')),
        size=(args.target_size, args.target_size),
        batch=1,
        source_transforms=[
            utils.transform.Resize(scale=.5),
            utils.transform.Resize(scale=2.),
        ]
    )

    model = SuperResolutionModel(shape=(args.target_size, args.target_size, 1))
    model.train(train_generator=train_generator,
                val_generator=val_generator,
                epochs=args.epoch,
                config=args)
    model.save('model.hdf5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Multi Object Tracking')
    parser.add_argument("command", metavar="<command>",
                        choices=['train', 'eval'],
                        help="'train' or 'eval'")
    parser.add_argument("--dataset", type=str, default='./data', required=False,
                        help="Dataset root directory")

    parser.add_argument("--epoch", type=int, default=10000, required=False,
                        help="Epoch for training")
    parser.add_argument("--interval", type=int, default=100, required=False)

    parser.add_argument("--source-size", type=int, default=16, required=False)
    parser.add_argument("--target-size", type=int, default=32, required=False)

    parser.add_argument("--lr", type=float, default=.001, required=False)
    parser.add_argument("--lr-decay", type=float, default=.0, required=False)

    parser.add_argument("--log", type=str, default='./logs', required=False,
                        help="Logging directory")
    parser.add_argument("--seed", type=int, default=42, required=False,
                        help="The answer to life the universe and everything")

    arguments = parser.parse_args()

    utils.init(arguments.seed)

    main(arguments)
