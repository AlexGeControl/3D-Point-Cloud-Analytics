#!/opt/conda/envs/kitti-detection-pipeline/bin/python

# extract.py
#     1. load label, calib and velodyne data from original KITTI 3D Object Detection dataset
#     2. extract target point cloud from velodyne measurements using label and calib parameters
#     3. save data in modelnet40 normal resampled format for later modelling

import argparse
import os 
import glob
import shutil

import struct
import progressbar
from random import sample 

import numpy as np
import pandas as pd
import open3d as o3d

import matplotlib.pyplot as plt

def read_velodyne_bin(filepath):
    '''
    Read Velodyne measurements from input bin file

    Parameters
    ----------
    filepath: str
        File path of Velodyne measurements as bin file

    Returns
    ----------
    point_cloud: numpy.ndarray
        Velodyne measurements as N-by-3 numpy ndarray

    '''
    point_cloud = []
    with open(filepath, 'rb') as f:
        # unpack velodyne frame:
        content = f.read()
        measurements = struct.iter_unpack('ffff', content)
        # parse:
        for i, point in enumerate(measurements):
            x, y, z, intensity = point
            point_cloud.append([x, y, z, intensity])
    # format for output
    point_cloud = np.asarray(point_cloud, dtype=np.float32)

    return point_cloud

def read_calib(filepath):
    '''
    Read KITTI 3D Object calibration data for frame transformations

    Parameters
    ----------
    filepath: str
        File path of KITTI 3D Object calibration data

    Returns
    ----------
    label: pandas.DataFrame
        KITTI 3D Object label as pandas.DataFrame

    '''
    DIMENSION = {
        'P0': (3, 4),
        'P1': (3, 4),
        'P2': (3, 4),
        'P3': (3, 4),
        'R0_rect': (3, 3),
        'Tr_velo_to_cam': (3, 4),
        'Tr_imu_to_velo': (3, 4)
    }

    param = {}
    # parse calibration data:
    with open(filepath, 'rt') as f:
        # one line per param:
        content = [tuple(i.split(':')) for i in f.read().strip().split('\n')]
        # format param as numpy.ndarray with correct shape
        for name, value in content:
            param[name] = np.asarray(
                [float(v) for v in value.strip().split()]
            ).reshape(
                DIMENSION[name]
            )
    
    return param

def transform_from_cam_to_velo(X_cam, param):
    '''
    Transform points from camera frame into velodyne frame

    Parameters
    ----------
    X_cam: numpy.ndarray
        Points in camera frame
    param: dict
        Vehicle parameters

    Returns
    ----------
    X_velo: numpy.ndarray
        Points in velodyne frame

    '''
    # get params:
    R0_rect = param['R0_rect']
    R_velo_to_cam, t_velo_to_cam = param['Tr_velo_to_cam'][:,0:3], param['Tr_velo_to_cam'][:,3]

    # unrectify:
    X_velo = np.dot(
        R0_rect.T, X_cam.T
    ).T

    # project to velo frame:
    X_velo = np.dot(
        R_velo_to_cam.T, (X_velo - t_velo_to_cam).T
    ).T

    return X_velo

def transform_from_velo_to_obj(X_velo, param, t_obj_to_cam, ry):
    '''
    Transform points from velodyne frame into object frame

    Parameters
    ----------
    X_velo: numpy.ndarray
        Points in velodyne frame
    param: dict
        Vehicle parameters
    t_obj_to_cam: numpy.ndarray
        Object center in camera frame
    ry:
        Object heading in camera frame

    Returns
    ----------
    X_obj: numpy.ndarray
        Points in object frame

    '''
    # get params:
    R0_rect = param['R0_rect']
    R_velo_to_cam, t_velo_to_cam = param['Tr_velo_to_cam'][:,0:3], param['Tr_velo_to_cam'][:,3]

    # project to unrectified camera frame:
    X_cam = np.dot(
        R_velo_to_cam, X_velo.T
    ).T + t_velo_to_cam

    # rectify:
    X_cam = np.dot(
       R0_rect, X_cam.T
    ).T

    # project to object frame:
    cos_ry = np.cos(ry)
    sin_ry = np.sin(ry)

    R_obj_to_cam = np.asarray(
        [
            [ cos_ry, 0.0, sin_ry],
            [    0.0, 1.0,    0.0],
            [-sin_ry, 0.0, cos_ry]
        ]
    )

    X_obj = np.dot(
        R_obj_to_cam.T, (X_cam - t_obj_to_cam).T
    ).T

    return X_obj

def filter_by_bouding_box(X, labels, dims):
    '''
    Filter point measurements using bounding box in object frame

    Parameters
    ----------
    X: numpy.ndarray
        Points in object frame
    labels: numpy.ndarray
        Point labels from segmentation
    dims: numpy.ndarray
        Bounding box dimensions

    Returns
    ----------
    object_id: int
        Dominant object id

    '''

    # filter by bounding box in object frame:
    idx_obj = np.all(
        np.logical_and(
            X >= -dims/2,
            X <=  dims/2
        ),
        axis = 1
    )

    if idx_obj.sum() == 0:
        return None

    # get object ID using non-maximum suppression:
    ids, counts = np.unique(
        labels[idx_obj], return_counts=True
    )
    object_id, _ = max(zip(ids, counts), key=lambda x:x[1]) 

    return object_id

def read_label(filepath, param):
    '''
    Read KITTI 3D Object label as Pandas DataFrame

    Parameters
    ----------
    filepath: str
        File path of KITTI 3D Object label

    Returns
    ----------
    label: pandas.DataFrame
        KITTI 3D Object label as pandas.DataFrame

    '''
    # load data:    
    df_label = pd.read_csv(
        filepath,
        sep = ' ', header=None
    )

    # add attribute names:
    df_label.columns = [
        'type',
        'truncated',
        'occluded',
        'alpha',
        'left', 'top', 'right', 'bottom',
        'height', 'width', 'length',
        'cx', 'cy', 'cz', 'ry'
    ]

    # filter label with no dimensions:
    condition = (
        (df_label['height'] >= 0.0) &
        (df_label['width'] >= 0.0) &
        (df_label['length'] >= 0.0)
    )
    df_label = df_label.loc[
        condition, df_label.columns
    ]

    #
    # get object center in velo frame:
    #
    centers_cam = df_label[['cx', 'cy', 'cz']].values
    centers_velo = transform_from_cam_to_velo(centers_cam, param)
    # add height bias:
    df_label['vx'] = df_label['vy'] = df_label['vz'] = 0.0
    df_label[['vx', 'vy', 'vz']] = centers_velo
    df_label['vz'] += df_label['height']/2

    # add radius for point cloud extraction:
    df_label['radius'] = df_label.apply(
        lambda x: np.linalg.norm(
            0.5*np.asarray(
                [x['height'], x['width'], x['length']]
            )
        ),
        axis = 1
    )

    return df_label

def init_label():
    """
    Get label template for extracted classification dataset

    Returns
    ----------
    labels: dict
        label container for extracted classification dataset

    """
    return {
        # original category:
        'type': [],
        'truncated': [],
        'occluded': [],
        # distance and num. of measurements:
        'vx': [], 
        'vy': [], 
        'vz': [], 
        'num_measurements': [],
        # bounding box labels:
        'height': [], 
        'width': [], 
        'length':[], 
        'ry':[]
    }

def add_label(dataset_label, category, label, N, center):
    """
    Add new label for extracted classification dataset

    Parameters
    ----------
    dataset_label: dict
        label container for extracted classification dataset
    category: str
        label of extracted object
    label: pandas.Series
        one KITTI 3D Object label record as pandas Series
    N: int
        number of Velodyne measurements
    center: numpy.ndarray
        point cloud center as numpy.ndarray

    Returns
    ----------
    None

    """
    if label is None:
        # kitti category:
        dataset_label[category]['type'].append('Unknown')
        dataset_label[category]['truncated'].append(-1)
        dataset_label[category]['occluded'].append(-1)

        # bounding box labels:
        dataset_label[category]['height'].append(-1)
        dataset_label[category]['width'].append(-1)
        dataset_label[category]['length'].append(-1)
        dataset_label[category]['ry'].append(-10)
    else:
        # kitti category:
        dataset_label[category]['type'].append(label['type'])
        dataset_label[category]['truncated'].append(label['truncated'])
        dataset_label[category]['occluded'].append(label['occluded'])

        # bounding box labels:
        dataset_label[category]['height'].append(label['height'])
        dataset_label[category]['width'].append(label['width'])
        dataset_label[category]['length'].append(label['length'])
        dataset_label[category]['ry'].append(label['ry'])

    # distance and num. of measurements:
    dataset_label[category]['num_measurements'].append(N)
    vx, vy, vz = center
    dataset_label[category]['vx'].append(vx)
    dataset_label[category]['vy'].append(vy)
    dataset_label[category]['vz'].append(vz)


def get_object_pcd_df(pcd, idx, N):
    """
    Format point cloud with normal as dataframe

    Parameters
    ----------
    pcd: open3d.PointCloud
        Velodyne measurements as Open3D PointCloud
    idx: numpy.ndarray
        object indices for point coordinates and surface normal extraction
    N: int
        number of Velodyne measurements of extracted object
    """
    df_point_cloud_with_normal = pd.DataFrame(
        data = np.hstack(
            (
                np.asarray(pcd.points)[idx],
                np.asarray(pcd.normals)[idx]
            )
        ),
        index = np.arange(N),
        columns = ['vx', 'vy', 'vz', 'nx', 'ny', 'nz']
    )

    return df_point_cloud_with_normal

def get_object_category(object_type):
    """
    Map KITTI 3D Object category to that of extracted classification dataset

    Parameters
    ----------
    object_type: str
        KITTI 3D Object category
    
    Returns
    ----------
    category: str
        corresponding object category in extracted classification dataset

    """
    category = 'vehicle'

    if object_type is None or object_type == 'Misc' or object_type == 'DontCare':
        category = 'misc'
    elif object_type == 'Pedestrian' or object_type == 'Person_sitting':
        category = 'pedestrian'
    elif object_type == 'Cyclist':
        category = 'cyclist'


    return category

def segment_ground_and_objects(point_cloud):
    """
    Segment ground plane and foreground objects from Velodyne measurement

    Parameters
    ----------
    point_cloud: numpy.ndarray
        Velodyne measurements as N-by-3 numpy.ndarray

    Returns
    ----------
    segmented_cloud: numpy.ndarray
        Segmented surrounding objects as N-by-3 numpy.ndarray
    segmented_ground: numpy.ndarray
        Segmented ground as N-by-3 numpy.ndarray

    """
    # TODO 01 -- ground segmentation
    N, _ = point_cloud.shape

    #
    # pre-processing: filter by surface normals
    #
    # first, filter by surface normal
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(point_cloud)
    pcd_original.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=5.0, max_nn=9
        )
    )

    # keep points whose surface normal is approximate to z-axis for ground plane segementation:
    normals = np.asarray(pcd_original.normals)
    angular_distance_to_z = np.abs(normals[:, 2])
    idx_downsampled = angular_distance_to_z > np.cos(np.pi/6)

    #
    # plane segmentation with RANSAC
    #
    # ground segmentation using PLANE RANSAC from PCL:
    pcd_downsampled = o3d.geometry.PointCloud()
    pcd_downsampled.points = o3d.utility.Vector3dVector(point_cloud[idx_downsampled])

    ground_model, idx_ground = pcd_downsampled.segment_plane(
        distance_threshold=0.30,
        ransac_n=3,
        num_iterations=1000
    )

    # 
    # post-processing: get ground output by distance to segemented plane
    # 
    segmented_ground = pcd_downsampled.select_by_index(idx_ground)

    distance_to_ground = np.abs(
        np.dot(point_cloud,np.asarray(ground_model[:3])) + ground_model[3]
    )
    idx_cloud = distance_to_ground > 0.30

    # limit FOV to front:
    segmented_objects = o3d.geometry.PointCloud()

    idx_segmented_objects = np.logical_and.reduce(
        [
            idx_cloud,
            point_cloud[:, 0] >=   1.95, point_cloud[:, 0] <=  80.00,
            point_cloud[:, 1] >= -30.00, point_cloud[:, 1] <= +30.00
        ]
    )

    segmented_objects.points = o3d.utility.Vector3dVector(
        point_cloud[idx_segmented_objects]
    )
    segmented_objects.normals = o3d.utility.Vector3dVector(
        np.asarray(pcd_original.normals)[idx_segmented_objects]
    )

    segmented_ground.paint_uniform_color([0.0, 0.0, 0.0])
    segmented_objects.paint_uniform_color([0.5, 0.5, 0.5])

    # foreground objects:
    labels = np.asarray(segmented_objects.cluster_dbscan(eps=0.60, min_points=3))

    return segmented_ground, segmented_objects, labels

def process_sample(index, input_dir, output_dir, dataset_label, debug):
    """
    Process one sample from KITTI 3D Object

    Parameters
    ----------
    index: int
        KITTI 3D Object measurement ID
    input_dir: str
        Root path of input KITTI 3D Object dataset
    output_dir: str
        Root path of extracted classification dataset
    dataset_label: dict
        label container for extracted classification dataset
    debug: bool
        debug mode selection. when activated the segmented object will be visualized using Open3D
    
    Returns
    ----------
    None

    """
    # load point cloud measurements:
    point_cloud = read_velodyne_bin(
        os.path.join(input_dir, 'velodyne', f'{index:06d}.bin')
    )
    # load calibration results:
    param = read_calib(
        os.path.join(input_dir, 'calib', f'{index:06d}.txt')
    )
    # load label:
    df_label = read_label(
        os.path.join(input_dir, 'label_2', f'{index:06d}.txt'),
        param
    )
    
    # segment ground and objects:
    segmented_ground, segmented_objects, object_ids = segment_ground_and_objects(point_cloud[:, 0:3])

    # build search tree on segmented object:
    search_tree = o3d.geometry.KDTreeFlann(segmented_objects)

    # identify segmented objects that have KITTI label:
    identified = set()
    for idx, label in df_label.iterrows():
        # parse params:
        center_velo = np.asarray([label['vx'], label['vy'], label['vz']])
        center_cam = np.asarray([label['cx'], label['cy'], label['cz']])
        # dimensions in camera frame:
        dims = np.asarray([label['length'], label['height'], label['width']])
        
        # identify the sphere of object using RNN:
        [k, idx, _] = search_tree.search_radius_vector_3d(
            center_velo, 
            label['radius']
        )

        # find object within label region:
        if (k > 0):     
            point_cloud_velo_ = np.asarray(segmented_objects.points)[idx]
            object_ids_ = object_ids[idx]

            # project to object frame for bounding box filtering:
            point_cloud_obj = transform_from_velo_to_obj(
                point_cloud_velo_, 
                param, 
                center_cam, 
                label['ry']
            )

            # add bias along height:
            point_cloud_obj[:, 1] += label['height']/2

            # get object id:
            object_id_ = filter_by_bouding_box(point_cloud_obj, object_ids_, dims)

            if object_id_ is None:
                continue
                
            identified.add(object_id_)

            # paint object:
            idx_object = np.asarray(idx)[object_ids_ == object_id_]

            if debug:
                # paint segmented objects as green:
                np.asarray(segmented_objects.colors)[
                    idx_object, :
                ] = [0.0, 1.0, 0.0]
            else:
                # format as pandas dataframe:
                N = len(idx_object)
                df_point_cloud_with_normal = get_object_pcd_df(segmented_objects, idx_object, N)

                # add label:
                category = get_object_category(label['type'])
                center = np.asarray(segmented_objects.points)[idx_object].mean(axis = 0)
                add_label(dataset_label, category, label, N, center)

                # write output:
                dataset_index = len(dataset_label[category]['type'])
                df_point_cloud_with_normal.to_csv(
                    os.path.join(output_dir, category, f'{dataset_index:06d}.txt'),
                    index=False, header=None
                )
    if debug:
        # visualize the segmented object on gray background:
        o3d.visualization.draw_geometries([segmented_ground, segmented_objects])
    else:
        # for segmented objects that don't have KITTI label, random sample and mark it as 'misc':
        unidentified = sample(
            [i for i in np.unique(object_ids) if i not in identified],
            3
        )
        for object_id_ in unidentified:
            idx_object = (object_ids == object_id_)
            N = idx_object.sum()
            df_point_cloud_with_normal = get_object_pcd_df(segmented_objects, idx_object, N)

            # add label:
            category = get_object_category(None)
            center = np.asarray(segmented_objects.points)[idx_object].mean(axis = 0)
            add_label(dataset_label, category, None, N, center)

            # write output:
            dataset_index = len(dataset_label[category]['type'])
            df_point_cloud_with_normal.to_csv(
                os.path.join(output_dir, category, f'{dataset_index:06d}.txt'),
                index=False, header=None
            )

def main(input_dir, output_dir, debug_index, num_limit):
    """
    Process KITTI 3D Object

    Parameters
    ----------
    input_dir: str
        Root path of input KITTI 3D Object dataset
    output_dir: str
        Root path of extracted classification dataset
    debug_index: int
        for value < 0, process the whole dataset
        otherwise process the selected instance in debug mode

    Returns
    ----------
    None

    """
    if debug_index < 0:
        N = len(
            glob.glob(
                os.path.join(input_dir, 'label_2', '*.txt')
            )
        )
        
        # init output root dir:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir) 
        os.makedirs(output_dir)

        # init output dir for each category:
        dataset_label = {}
        for category in ['vehicle', 'pedestrian', 'cyclist', 'misc']:
            os.makedirs(os.path.join(output_dir, category))
            dataset_label[category] = init_label()

        # init error list:
        index_errors = []

        for index in progressbar.progressbar(
            range(N if (num_limit is None) else min(num_limit, N))
        ):
            try:
                process_sample(index, input_dir, output_dir, dataset_label, False)
            except:
                print(f'[KITTI Object Classification Dataset ETL]: Failed to process {index:06d}.txt')
                index_errors.append(index)

        # write metadata for feature engineering:
        for category in ['vehicle', 'pedestrian', 'cyclist', 'misc']:
            dataset_label[category] = pd.DataFrame.from_dict(
                dataset_label[category]
            )
            dataset_label[category].to_csv(
                os.path.join(output_dir, f'{category}.txt'),
                index=False
            )
        
        print('[KITTI Object Classification Dataset ETL]: Failed to process the measurements below. Please inspect later.')
        for index in index_errors:
            print(f'\t{index:06d}.txt')
    else:
        process_sample(debug_index, input_dir, None, None, True)


def get_arguments():
    """ 
    Get command-line arguments for KITTI 3D Object ETL

    """
    # init parser:
    parser = argparse.ArgumentParser("Extract point cloud from KITTI 3D Object Detection for object classification.")

    # add required and optional groups:
    required = parser.add_argument_group('Required')
    optional = parser.add_argument_group('Optional')

    # add required:
    required.add_argument(
        "-i", dest="input", help="Input path of KITTI 3D Object dataset.",
        required=True
    )
    required.add_argument(
        "-o", dest="output", help="Output path of generated classification dataset.",
        required=True
    )

    optional.add_argument(
        "-d", dest="debug_index", help="Sample index for debugging. Defaults to -1",
        required=False, type=int, default=-1
    )
    optional.add_argument(
        "-l", dest="num_limit", help="Maximum number of samples to be processed. Defaults to None.",
        required=False, type=int, default=None
    )

    # parse arguments:
    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments:
    arguments = get_arguments()

    # get dataset metadata:
    main(
        arguments.input, arguments.output,
        arguments.debug_index, arguments.num_limit
    )

