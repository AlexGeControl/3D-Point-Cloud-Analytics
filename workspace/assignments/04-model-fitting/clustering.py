#!/opt/conda/envs/point-cloud/bin/python

# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import argparse

import os
import glob
import random

import struct

import numpy as np
# Open3D:
import open3d as o3d
# PCL utils:
import pcl
from utils.segmenter import GroundSegmenter
# sklearn:
from sklearn.cluster import DBSCAN

from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)


def ground_segmentation(data):
    """
    Segment ground plane from Velodyne measurement

    Parameters
    ----------
    data: numpy.ndarray
        Velodyne measurements as N-by-3 numpy.ndarray

    Returns
    ----------
    segmented_cloud: numpy.ndarray
        Segmented surrounding objects as N-by-3 numpy.ndarray
    segmented_ground: numpy.ndarray
        Segmented ground as N-by-3 numpy.ndarray

    """
    # TODO 01 -- ground segmentation
    N, _ = data.shape

    #
    # pre-processing: filter by surface normals
    #
    # first, filter by surface normal
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(data)
    pcd_original.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=5.0, max_nn=9
        )
    )

    # keep points whose surface normal is approximate to z-axis for ground plane segementation:
    normals = np.asarray(pcd_original.normals)
    angular_distance_to_z = np.abs(normals[:, 2])
    idx_downsampled = angular_distance_to_z > np.cos(np.pi/6)
    downsampled = data[idx_downsampled]

    #
    # plane segmentation with RANSAC
    #
    # ground segmentation using PLANE RANSAC from PCL:
    cloud = pcl.PointCloud()
    cloud.from_array(downsampled)
    ground_segmenter = GroundSegmenter(cloud=cloud)
    inliers, model = ground_segmenter.segment()

    # 
    # post-processing: get ground output by distance to segemented plane
    # 
    distance_to_ground = np.abs(
        np.dot(data,np.asarray(model[:3])) + model[3]
    )

    idx_ground = distance_to_ground <= ground_segmenter.get_max_distance()
    idx_segmented = np.logical_not(idx_ground)

    segmented_cloud = data[idx_segmented]
    segmented_ground = data[idx_ground]

    print(
        f'[Ground Segmentation]: \n\tnum. origin measurements: {N}\n\tnum. segmented cloud: {segmented_cloud.shape[0]}\n\tnum. segmented ground: {segmented_ground.shape[0]}\n'
    )
    return segmented_cloud, segmented_ground


def clustering(data):
    """
    Segment surrounding objects using DBSCAN

    Parameters
    ----------
    data: numpy.ndarray
        Segmented point cloud as N-by-3 numpy.ndarray

    Returns
    ----------
    cluster_index: list of int
        Cluster ID for each point

    """
    # TODO 02 -- surrounding object segmentation
    cluster_index = DBSCAN(
        eps=0.25, min_samples=5, n_jobs=-1
    ).fit_predict(data)

    return cluster_index


def plot_clusters(segmented_ground, segmented_cloud, cluster_index):
    """
    Visualize segmentation results using Open3D

    Parameters
    ----------
    segmented_cloud: numpy.ndarray
        Segmented surrounding objects as N-by-3 numpy.ndarray
    segmented_ground: numpy.ndarray
        Segmented ground as N-by-3 numpy.ndarray
    cluster_index: list of int
        Cluster ID for each point

    """
    def colormap(c, num_clusters):
        """
        Colormap for segmentation result

        Parameters
        ----------
        c: int 
            Cluster ID
        C

        """
        # outlier:
        if c == -1:
            color = [1]*3
        # surrouding object:
        else:
            color = [0] * 3
            color[c % 3] = c/num_clusters

        return color

    # ground element:
    pcd_ground = o3d.geometry.PointCloud()
    pcd_ground.points = o3d.utility.Vector3dVector(segmented_ground)
    pcd_ground.colors = o3d.utility.Vector3dVector(
        [
            [0.372]*3 for i in range(segmented_ground.shape[0])
        ]
    )

    # surrounding object elements:
    pcd_objects = o3d.geometry.PointCloud()
    pcd_objects.points = o3d.utility.Vector3dVector(segmented_cloud)
    num_clusters = max(cluster_index) + 1
    pcd_objects.colors = o3d.utility.Vector3dVector(
        [
            colormap(c, num_clusters) for c in cluster_index
        ]
    )

    # visualize:
    o3d.visualization.draw_geometries([pcd_ground, pcd_objects])  

def get_arguments():
    """ 
    Get command-line arguments

    """
    # init parser:
    parser = argparse.ArgumentParser("Perform ground & surrounding object segmentation on KITTI 3D Object Detection.")

    # add required and optional groups:
    required = parser.add_argument_group('Required')

    # add required:
    required.add_argument(
        "-i", dest="input", help="Input path of velodyne point cloud.",
        required=True
    )
    required.add_argument(
        "-n", dest="num_cases", help="The number of samples.",
        required=True, type=int
    )

    # parse arguments:
    return parser.parse_args()

def main(input_dir, num_cases):
    pattern = os.path.join(input_dir, '*.bin')
    samples = random.sample(glob.glob(pattern), num_cases)

    print('KITTI 3D Object Detection Pipeline')
    for sample in samples:
        print(f'\t Process {sample} ...')

        # read Velodyne measurements:
        lidar_measurements = read_velodyne_bin(sample)
        # segment ground:
        segmented_cloud, segmented_ground = ground_segmentation(data=lidar_measurements)
        # segment surrouding objects:
        cluster_index = clustering(segmented_cloud)

        # visualize with Open3D:
        plot_clusters(segmented_ground, segmented_cloud, cluster_index)


if __name__ == '__main__':
    # parse arguments:
    arguments = get_arguments()

    # get input point cloud filename:
    main(arguments.input, arguments.num_cases)