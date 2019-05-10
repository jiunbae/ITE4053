import argparse

from utils import init


def main(args: argparse.Namespace):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Multi Object Tracking')
    parser.add_argument("--dataset", type=str, default='../datasets/MOT17/train',
                        help="train data path")
    parser.add_argument("--dest", type=str, default='./results',
                        help="result destination")
    parser.add_argument("--support", type=str, default=None,
                        help="Support detection")
    parser.add_argument("--support-only", action='store_true', default=False,
                        help="Support detection only")
    parser.add_argument("--type", type=str, default='MOT',
                        help="Dataset type, default=MOT")

    parser.add_argument("--cache", action='store_true', default=False,
                        help="Use previous results for evaluation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Manual seed")

    arguments = parser.parse_args()

    init(arguments.seed)

    main(arguments)
