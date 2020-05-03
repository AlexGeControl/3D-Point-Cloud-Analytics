#!/opt/conda/envs/point-cloud/bin/python

import argparse
import os 
import pathlib
import shutil
import glob
from random import sample

def get_arguments():
    """ gets command line arguments.
    :return:
    """

    # init parser:
    parser = argparse.ArgumentParser("Downsample ModelNet40 by category.")

    # add required and optional groups:
    required = parser.add_argument_group('Required')
    optional = parser.add_argument_group('Optional')

    # add required:
    required.add_argument(
        "-n", dest="num_per_category", help="The number of samples per category.",
        required=True, type=int
    )

    # add optional:
    optional.add_argument(
        "-i", dest="input", help="Input path of original ModelNet 40 dataset. Defaults to $PWD/ModelNet40",
        default='./ModelNet40'
    )
    optional.add_argument(
        "-t", dest="type", help="Which subset to dowmsample from. Defaults to train",
        default='train'
    )
    optional.add_argument(
        "-o", dest="output", help="Output path of downsampled ModelNet 40 dataset. Defaults to $PWD/ModelNet40Downsampled",
        default='./ModelNet40Downsampled'
    )    

    # parse arguments:
    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments:
    arguments = get_arguments()

    print(f'Downsample ModelNet40 -- {arguments.num_per_category} per category: Start ...')

    # create output dir:
    try:
        pathlib.Path(arguments.output).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        shutil.rmtree(arguments.output)
        pathlib.Path(
            os.path.join(arguments.output, 'off')
        ).mkdir(parents=True, exist_ok=False)
        pathlib.Path(
            os.path.join(arguments.output, 'ply')
        ).mkdir(parents=True, exist_ok=False)

    # sample from input dir:
    for category in os.listdir(arguments.input):
        # identify dataset root dir of given category:
        input_dir = os.path.join(arguments.input, category, arguments.type)
        # find all *.off and downsample:
        pattern = os.path.join(input_dir, '*.off')
        samples = sample(glob.glob(pattern), arguments.num_per_category)
        # move the samples into output dir:
        print(f'\t{category}')
        for src in samples:
            dst = os.path.join(
                arguments.output,
                'off',
                os.path.basename(src)
            )
            shutil.copyfile(src, dst)

            print(f'\t\t{src} --> {dst}')

    print(f'Downsample ModelNet40 -- {arguments.num_per_category} per category: Done.')