import argparse


parser = argparse.ArgumentParser(description='ITE4053 Assignment 1'
                                             'Neural Network implemented by numpy and TensorFlow')

parser.add_argument('--mode', type=str, default='tf',
                    choices=['np', 'tf'], required=True,
                    help='Select mode for assignment')

parser.add_argument('--epoch', type=int, default=5000,
                    help="Epoch size")
parser.add_argument('--size', type=int, default=128,
                    help="Dataset size")
parser.add_argument('--lr', type=float, default=.1,
                    help="Learning rate")
parser.add_argument('--optimizer', type=str, default='SGD',
                    choices=['SGD', 'Adam'],
                    help="Optimizer")
parser.add_argument('--loss', type=str, default='binary_crossentropy',
                    choices=['binary_crossentropy', 'mean_squared_error'],
                    help="Loss function")
parser.add_argument('--normal', action='store_true',
                    help='Use normalized dataset')
parser.add_argument('-l', '--layer', action='append', nargs='*',
                    help='Define Layer (input_dim, output_dim, activation)'
                         'Activation must be one of {sigmoid, relu}')
parser.add_argument('--save', type=str, default='parameters.npz',
                    help='Save parameter path (only works on np mod)')
parser.add_argument('--repeat', type=int, default=10,
                    help="Repeat train, valid")
parser.add_argument('--seed', type=int, default=2,
                    help="Manual seed")

arguments = parser.parse_args()
