#!/opt/conda/envs/kitti-detection-pipeline/bin/python

# generate-training-set.py
#     1. To be added
#     2. To be added
#     3. To be added


import argparse

import os 
import glob
import re

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from palettable.colorbrewer.qualitative import Pastel1_7


def load_labels(input_dir):
    """
    Load labels from the extracted object classification dataset

    Parameters
    ----------
    input_dir: str
        input path of extracted object classification dataset.

    Returns
    ----------
    labels: dict of pandas.DataFrame
        labels of the extracted dataset

    """
    # visualize dataset metadata:
    labels = {}

    # load labels:
    filename_labels = [
        f for f in glob.glob(
            os.path.join(input_dir, '*.txt')
         ) if re.search(r'(cyclist|misc|pedestrian|vehicle)\.txt$', f)
    ]
    for filename_label in filename_labels:
        # get current category:
        category = os.path.splitext(
            os.path.basename(filename_label)
        )[0]
        
        # load label:
        labels[category] = pd.read_csv(filename_label)

    return labels

def draw_class_distribution(labels):
    """
    Visualize class distribution

    Parameters
    ----------
    labels: dict of pandas.DataFrame
        labels of the extracted dataset

    Returns
    ----------
    None

    """
    # generate category counts:
    categories, counts = zip(
        *[
            (category.upper(), labels[category].shape[0]) for category in labels
        ]
    )

    # convert counts to percentages:
    counts = np.asarray(counts)
    percentages = counts / counts.sum()
    categories = [
        f'{c}, {100*p:.2f}%' for (c, p) in zip(categories, percentages)
    ]

    # draw the plot:
    plt.figure(num=None, figsize=(16, 9))
    plt.pie(counts, labels=categories, colors=Pastel1_7.hex_colors)
    p=plt.gcf()
    p.gca().add_artist(
        plt.Circle(
            (0,0), 0.7, color='white'
        )
    )
    plt.title('Class Distribution, Segmented Objects')
    plt.show()

def draw_measurement_count(labels):
    """
    Visualize the relationship between measurement counts and object-ego vehicle distance

    """
    df_distance_count = []

    for category in labels:
        df_ = labels[category].loc[:, ['num_measurements']]
        df_['distance'] = np.sqrt(labels[category]['vx']**2 + labels[category]['vy']**2)
        df_['category'] = category

        df_distance_count.append(df_)
    
    df_distance_count = pd.concat(df_distance_count)

    # reduce resolution:
    df_distance_count['distance'] = df_distance_count['distance'].apply(np.round)

    # overview:
    plt.figure(num=None, figsize=(16, 9))

    sns.lineplot(
        x="distance", y="num_measurements",
        hue="category", style="category",
        markers=True, dashes=False, data=df_distance_count
    )

    plt.title('Num. Measurements as a function of Object Distance')
    plt.show()

    # ROI selection:
    plt.figure(num=None, figsize=(16, 9))

    sns.lineplot(
        x="distance", y="num_measurements",
        hue="category", style="category",
        markers=True, dashes=False, data=df_distance_count.loc[
            (df_distance_count["distance"] >= 20.0) & (df_distance_count["distance"] <= 45.0),
            df_distance_count.columns
        ]
    )

    plt.title('Num. Measurements as a function of Object Distance, ROI Selection')
    plt.show()

def get_arguments():
    """ 
    Get command-line arguments for object classification training set generation.

    """
    # init parser:
    parser = argparse.ArgumentParser("Generate training set for pointnet++ object classification.")

    # add required and optional groups:
    required = parser.add_argument_group('Required')
    optional = parser.add_argument_group('Optional')

    # add required:
    required.add_argument(
        "-i", dest="input", help="Input path of extracted object classification dataset.",
        required=True
    )

    optional.add_argument(
        "-m", dest="mode", help="Running mode. 'analyze' for dataset analytics and 'generate' for generation. Defaults to 'analyze'",
        required=False, type=str, default="analyze"
    )

    # parse arguments:
    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments:
    arguments = get_arguments()

    # load labels
    labels = load_labels(
        arguments.input
    )

    # visualization 01: category distribution
    draw_class_distribution(labels)

    # visualization 02: distance -- measurement count
    draw_measurement_count(labels)