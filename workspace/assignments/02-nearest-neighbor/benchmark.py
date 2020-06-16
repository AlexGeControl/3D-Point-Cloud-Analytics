#!/opt/conda/envs/02-nearest-neighbor/bin/python

# 对数据集中的点云，批量执行构建树和查找，包括kdtree和octree，并评测其运行时间

import os, argparse, time
import struct

import random
import math
import numpy as np

from octree import OCTree
from kdtree import KDTree
from result_set import KNNResultSet, RadiusNNResultSet

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
    return np.asarray(pc_list, dtype=np.float32).T

def get_arguments():
    """ Get command-line arguments
    """
    # init parser:
    parser = argparse.ArgumentParser("KDTree & OcTree benchmark on KITTI.")

    # add required and optional groups:
    required = parser.add_argument_group('Required')

    # add required:
    required.add_argument(
        "-i", dest="input", help="Input path of Velodyne lidar measurements.",
        required=True
    )

    # parse arguments:
    return parser.parse_args()

def main(data_dir):
    # kNN & RNN configuration:
    leaf_size = 32
    min_extent = 0.0001
    k = 8
    radius = 1

    # kitti velodyne
    cat = os.listdir(data_dir)
    iteration_num = len(cat)

    print("OCTree --------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    for i in range(iteration_num):
        filename = os.path.join(data_dir, cat[i])
        point_cloud = read_velodyne_bin(filename)

        # build tree:
        begin_t = time.time()
        octree = OCTree(point_cloud = point_cloud, leaf_size = leaf_size, min_extent = min_extent)
        construction_time_sum += time.time() - begin_t

        query = point_cloud[0,:]

        # kNN query:
        begin_t = time.time()
        knn_result_set = KNNResultSet(capacity=k)
        octree.knn_search(query, knn_result_set)
        knn_time_sum += time.time() - begin_t

        # RNN query:
        begin_t = time.time()
        rnn_result_set = RadiusNNResultSet(radius=radius)
        octree.rnn_fast_search(query, rnn_result_set)
        radius_time_sum += time.time() - begin_t

        # brute force:
        begin_t = time.time()
        diff = np.linalg.norm(point_cloud - query, axis=1)
        nn_idx = np.argsort(diff)
        nn_dist = diff[nn_idx]
        brute_time_sum += time.time() - begin_t

    print(
        "Octree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (
            construction_time_sum*1000/iteration_num,
            knn_time_sum*1000/iteration_num,
            radius_time_sum*1000/iteration_num,
            brute_time_sum*1000/iteration_num
        )
    )

    print("KDTree --------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    for i in range(iteration_num):
        filename = os.path.join(data_dir, cat[i])
        point_cloud = read_velodyne_bin(filename)

        # build tree:
        begin_t = time.time()
        kd_tree = KDTree(point_cloud = point_cloud, leaf_size = leaf_size)
        construction_time_sum += time.time() - begin_t

        query = point_cloud[0,:]

        # kNN query:
        begin_t = time.time()
        knn_result_set = KNNResultSet(capacity=k)
        kd_tree.knn_search(query, knn_result_set)
        knn_time_sum += time.time() - begin_t

        # RNN query:
        begin_t = time.time()
        rnn_result_set = RadiusNNResultSet(radius=radius)
        kd_tree.rnn_search(query, rnn_result_set)
        radius_time_sum += time.time() - begin_t

        # brute force:
        begin_t = time.time()
        diff = np.linalg.norm(point_cloud - query, axis=1)
        nn_idx = np.argsort(diff)
        nn_dist = diff[nn_idx]
        brute_time_sum += time.time() - begin_t

    print(
        "Kdtree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (
            construction_time_sum * 1000 / iteration_num,
            knn_time_sum * 1000 / iteration_num,
            radius_time_sum * 1000 / iteration_num,
            brute_time_sum * 1000 / iteration_num
        )
    )


if __name__ == '__main__':
    # get command-line arguments:
    args = get_arguments()

    main(args.input)