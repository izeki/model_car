"""Command line arguments parser configuration."""
import argparse  # default python library for command line argument parsing
import os

parser = argparse.ArgumentParser(  # pylint: disable=invalid-name
    description='Train DNNs on model car data.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# parser.add_argument('--gpu', default=0, type=int, help='Cuda GPU ID, \
#                        not support, leave for keras to assign GPU resource')
parser.add_argument('--batch-size', default=100, type=int)
parser.add_argument('--display', dest='display', action='store_true')
parser.add_argument('--no-display', dest='display', action='store_false')
parser.set_defaults(display=True)

parser.add_argument('--network-model', default='SqueezeNet', type=str,
                    help='Network Model')

parser.add_argument('--verbose', default=True, type=bool,
                    help='Debugging mode')
parser.add_argument('--interactive', default=False, type=bool,
                    help='Interactive mode')
# parser.add_argument('--aruco', default=True, type=bool, help='Use Aruco data')
parser.add_argument('--data-path', default='/home/bdd/' 
                    + 'Desktop/training_data', type=str)
parser.add_argument('--resume-path', default='/home/bdd/' 
                    + 'Desktop/tmp/z2_color_squeeze_net_snap.hdf5',
                    type=str, help='Path to'
                    + ' resume file containing network state dictionary')
parser.add_argument('--save-path', default='/home/bdd/' 
                    + 'Desktop/tmp', type=str, help='Path to'
                    + ' folder to save net state dictionaries.')

# nargs='+' allows for multiple arguments and stores arguments in a list
parser.add_argument(
    '--ignore',
    default=(
        'reject_run',
        'left',
        'out1_in2',
        'play',
        'racing'),
    type=str,
    nargs='+',
    help='Skips these labels in data.')

parser.add_argument('--require-one', default=(), type=str, nargs='+',
                    help='Skips data without these labels in data.')
parser.add_argument('--use-states', default=(1, 3, 5, 6, 7), type=str,
                    nargs='+', help='Skips data outside of these states.')

parser.add_argument('--nframes', default=2, type=int,
                    help='# timesteps of camera input')
parser.add_argument('--nsteps', default=10, type=int,
                    help='# of steps of time to predict in the future')
parser.add_argument('--nstateframes', default=30, type=int,
                    help='# of timesteps of camera_input to test the quality of time synchronization')
parser.add_argument('--stride', default=3, type=int,
                    help="number of timesteps between network predictions")

parser.add_argument('--print-moments', default=1000, type=int,
                    help='# of moments between printing stats')

ARGS = parser.parse_args()

# Check for $DISPLAY being blank
if 'DISPLAY' not in os.environ:
    ARGS.display = False
