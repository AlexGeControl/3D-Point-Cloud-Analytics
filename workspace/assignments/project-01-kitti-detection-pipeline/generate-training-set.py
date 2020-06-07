#!/opt/conda/envs/kitti-detection-pipeline/bin/python

# generate-training-set.py
#     1. To be added
#     2. To be added
#     3. To be added


import argparse

import os 
import glob
import shutil
import re

import progressbar
import numpy as np
import scipy
import pandas as pd
import open3d as o3d
from sklearn.model_selection import StratifiedShuffleSplit

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

def draw_class_distribution(labels, object_type):
    """
    Visualize class distribution

    Parameters
    ----------
    labels: dict of pandas.DataFrame
        labels of the extracted dataset
    object_type: str
        object type for graph title display

    Returns
    ----------
    None

    """
    # generate category counts:
    categories, counts = zip(
        *[
            (
                category.upper(), 
                labels[category] if type(labels[category]) is int else labels[category].shape[0]
            ) for category in labels
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
    plt.title(f'Class Distribution, {object_type} Objects')
    plt.show()

def draw_measurement_count(labels):
    """
    Visualize the relationship between measurement counts and object-ego vehicle distance

    Parameters
    ----------
    labels: dict of pandas.DataFrame
        labels of the extracted dataset

    Returns
    ----------
    None

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

def preprocess(point_cloud_with_normal, num_sample_points, yaw=None, debug=False):
    """
    Preprocess point cloud with normal:
        1. Shift to zero-centered;
        2. Downsample / upsample to specified size;
        3. (Optional): rotate along z-axis

    Parameters
    ----------
    point_cloud_with_normal: numpy.ndarray
        point cloud with normal as N-by-6 numpy.ndarray
    num_sample_points: int
        standard point cloud size
    yaw: float
        rotation along z-axis. Defauts to None
    debug: bool
        debug mode selection. when activated the processed point cloud will be visualized using Open3D

    Returns
    ----------
    df_preprocessed_point_cloud_with_normal: pandas.DataFrame
        preprocessed point cloud with normal as pandas.DataFrame

    """
    # parse point coordinates and normals:
    N, _ = point_cloud_with_normal.shape
    points_original, normals_original = point_cloud_with_normal[:, 0:3], point_cloud_with_normal[:, 3:]

    # random sample according to distance:
    weights = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(points_original, 'euclidean')
    ).mean(axis = 0)
    weights /= weights.sum()
    idx = np.random.choice(
        np.arange(N), 
        size = (num_sample_points, ), replace=True if num_sample_points > N else False,
        p = weights
    )

    # translate to zero-mean:
    points_processed, normals_processed = points_original[idx], normals_original[idx]
    points_processed -= points_original.mean(axis = 0)

    if not (yaw is None):
        R = o3d.geometry.PointCloud.get_rotation_matrix_from_axis_angle(
            yaw * np.asarray([0.0, 0.0, 1.0])
        )
        points_processed = np.dot(points_processed, R.T)
        normals_processed = np.dot(normals_processed, R.T)

    if debug:
        # original:
        pcd_original = o3d.geometry.PointCloud()
        pcd_original.points = o3d.utility.Vector3dVector(points_original)
        pcd_original.normals = o3d.utility.Vector3dVector(normals_original)
        pcd_original.paint_uniform_color([1.0, 0.0, 0.0])
        # processed:
        pcd_processed = o3d.geometry.PointCloud()
        pcd_processed.points = o3d.utility.Vector3dVector(points_processed)
        pcd_processed.normals = o3d.utility.Vector3dVector(normals_processed)
        pcd_processed.paint_uniform_color([0.0, 1.0, 0.0])
        # draw:
        o3d.visualization.draw_geometries(
            [pcd_original, pcd_processed]
        )

    # format as numpy.ndarray:
    N, _ = points_processed.shape
    df_preprocessed_point_cloud_with_normal = pd.DataFrame(
        data = np.hstack(
            (points_processed, normals_processed)
        ),
        index = np.arange(N),
        columns = ['vx', 'vy', 'vz', 'nx', 'ny', 'nz']
    )

    return df_preprocessed_point_cloud_with_normal

def generate_training_set(labels, input_dir, output_dir, max_radius_distance, num_sample_points):
    """
    Generate training set

    Parameters
    ----------
    labels: dict of pandas.DataFrame
        labels of the extracted dataset
    input_dir: str
        Root path of original extracted classification dataset
    output_dir: str
        Root path of resampled training set
    max_radius_distance: float
        Maximum radius distance between object and Velodyne lidar
    num_sample_points: int
        Number of sample points to keep for each object

    Returns
    ----------
    None

    """
    # init output root dir:
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir) 
    os.makedirs(output_dir)

    for category in labels:
        os.makedirs(os.path.join(output_dir, category))

    #
    # stage 01: filter by radius distance
    # 
    counts = {}
    for category in labels:
        # restore filename of associated data:
        labels[category]['filename'] = np.asarray(
            [f'{i:06d}.txt' for i in np.arange(1, labels[category].shape[0] + 1)]
        )  
        # calculate radius distance
        labels[category]['distance'] = np.sqrt(labels[category]['vx']**2 + labels[category]['vy']**2)
        # filter by maximum radius distance:
        labels[category] = labels[category].loc[
            labels[category]['distance'] <= max_radius_distance,
            ['type', 'distance', 'num_measurements', 'filename']
        ]
        # generate class distribution counts for balancing through resample:
        counts[category] = labels[category].shape[0]

    # summary for stage 01:
    print('[KITTI Object Classification Dataset Generation]: Class distribution after ROI filtering')
    total = np.sum(
        list(counts.values())
    )
    for category in counts:
        print(f'\t{category.upper()}: {100.0 * counts[category] / total:.2f}% @ {counts[category]}')

    #
    # stage 02: resample
    # 
    for category in labels:
        # for each class, its up-sampling ratio is determined by its count with respect to that of 'misc':
        up_sampling_ratio = int(
            np.ceil(counts['misc'] / counts[category])
        )
        # reset count:
        counts[category] = 0
        # load data from the extracted dataset:
        for i, r in progressbar.progressbar(
            labels[category].iterrows()
        ):
            pcd_with_normal = pd.read_csv(
                os.path.join(input_dir, category, r['filename'])
            ).values

            # ignore hard case:
            N, _ = pcd_with_normal.shape
            if N <= 3:
                continue

            # for each object measurement, preprocess it:
            for _ in ([None] if up_sampling_ratio <= 1 else ([None] + list(range(up_sampling_ratio)))):
                # random yaw between [-np.pi/4, +np.pi/4]:
                yaw = np.pi / 4.0 * (2 * np.random.sample() - 1.0)
                df_preprocessed_point_cloud_with_normal = preprocess(pcd_with_normal, num_sample_points, yaw, False)

                # save:
                df_preprocessed_point_cloud_with_normal.to_csv(
                    os.path.join(output_dir, category, f'{counts[category]:06d}.txt'),
                    index=False, header=None
                )

                # update count:
                counts[category] += 1

    # summary for stage 01:
    print('[KITTI Object Classification Dataset Generation]: Class distribution after resampling')
    total = np.sum(
        list(counts.values())
    )
    for category in counts:
        print(f'\t{category.upper()}: {100.0 * counts[category] / total:.2f}% @ {counts[category]}')

    return counts

def generate_train_test_split(input_dir):
    """
    Generate train-test split

    Parameters
    ----------
    input_dir: str
        Root path of resampled training set

    Returns
    ----------
    None

    """
    # init:
    X = []
    y = []

    regex_parser = re.compile(r'.+/([a-z]+)/(\d{6})\.txt$')
    for filename in glob.glob(
        os.path.join(input_dir, '*', '*.txt')
    ):  
        # parse fields:
        fields = regex_parser.match(filename)
        category, idx = fields.group(1), fields.group(2)
        
        # add:
        X.append(f'{category}_{idx}')
        y.append(category)
    
    # format:
    X = np.asarray(X)
    y = np.asarray(y)

    # generate train-test split using stratified shuffle split:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    sss.get_n_splits(X, y)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]

    # write split:
    y_unique = np.unique(y)
    pd.DataFrame(
        data = y_unique,
        index = np.arange(len(y_unique)),
        columns = ['id']
    ).to_csv(
        os.path.join(input_dir, 'object_names.txt'),
        index=False, header=None
    )
    pd.DataFrame(
        data = X_train,
        index = np.arange(len(X_train)),
        columns = ['id']
    ).to_csv(
        os.path.join(input_dir, 'train.txt'),
        index=False, header=None
    )
    pd.DataFrame(
        data = X_test,
        index = np.arange(len(X_test)),
        columns = ['id']
    ).to_csv(
        os.path.join(input_dir, 'test.txt'),
        index=False, header=None
    )

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

    # add optional:
    optional.add_argument(
        "-m", dest="mode", help="Running mode. 'analyze' for dataset analytics and 'generate' for generation. Defaults to 'analyze'",
        required=False, type=str, choices=['analyze', 'generate'], default="analyze"
    )
    optional.add_argument(
        "-r", dest="max_radius_distance", help="Maximum radius distance between object and Velodyne lidar. \nUsed for ROI definition. Defaults to 25.0. \nONLY used in 'generate' mode.",
        required=False, type=float, default=25.0
    )
    optional.add_argument(
        "-n", dest="num_sample_points", help="Number of sample points to keep for each object. \nDefaults to 64. \nONLY used in 'generate' mode.",
        required=False, type=int, default=64
    )
    optional.add_argument(
        "-o", dest="output", help="Output path of generated training dataset. \nDefaults to current working directory. \nONLY used in 'generate' mode.",
        required=False, type=str, default="."
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

    #
    # mode 01: analyze dataset metadata
    # 
    if arguments.mode == 'analyze':
        # visualization 01: category distribution
        draw_class_distribution(labels, object_type='Segmented')

        # visualization 02: distance -- measurement count
        draw_measurement_count(labels)
    #
    # mode 02: generate training set for deep network
    # 
    elif arguments.mode == 'generate':
        # resample:
        counts = generate_training_set(
            labels,
            arguments.input, arguments.output,
            arguments.max_radius_distance, arguments.num_sample_points
        )

        # visualize class distribution on the resampled dataset:
        draw_class_distribution(counts, object_type='Resampled')

        # generate train-test split:
        generate_train_test_split(arguments.output)
    #   
    # mode otherwise: the program should never reach here
    # 
    else:
        print(f'[KITTI Object Classification Dataset Generation]: Invalid mode {arguments.mode}')