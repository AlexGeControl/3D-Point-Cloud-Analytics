#!/opt/conda/envs/07-feature-detection/bin/python

# detect-iss.py
#     1. load point cloud in modelnet40 normal format
#     2. calculate ISS keypoints
#     3. visualize the results


import argparse

import numpy as np
import pandas as pd
import open3d as o3d
import heapq

def read_modelnet40_normal(filepath):
    '''
    Read ModelNet40 sample with normal as Open3D point cloud

    Parameters
    ----------
    filepath: str
        File path of ModelNet40 sample

    Returns
    ----------
    point_cloud: numpy.ndarray
        Velodyne measurements as N-by-3 numpy ndarray

    '''
    # load data:
    df_point_cloud_with_normal = pd.read_csv(
        filepath, header=None
    )
    # add colunm names:
    df_point_cloud_with_normal.columns = [
        'x', 'y', 'z',
        'nx', 'ny', 'nz'
    ]
    
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        df_point_cloud_with_normal[['x', 'y', 'z']].values
    )
    pcd.normals = o3d.utility.Vector3dVector(
        df_point_cloud_with_normal[['nx', 'ny', 'nz']].values
    )

    return pcd


def get_arguments():
    """ 
    Get command-line arguments

    """
    # init parser:
    parser = argparse.ArgumentParser("Detect ISS keypoints on ModelNet40 dataset.")

    # add required and optional groups:
    required = parser.add_argument_group('Required')
    optional = parser.add_argument_group('Optional')

    # add required:
    required.add_argument(
        "-i", dest="input", help="Input path of ModelNet40 sample.",
        required=True
    )
    required.add_argument(
        "-r", dest="radius", help="Radius for radius nearest neighbor definition.",
        required=True, type=float
    )

    # parse arguments:
    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments:
    arguments = get_arguments()

    input_filename = arguments.input
    radius = arguments.radius

    # load point cloud
    pcd = read_modelnet40_normal(input_filename)

    # build search tree:
    search_tree = o3d.geometry.KDTreeFlann(pcd)

    # point handler:
    points = np.asarray(pcd.points)

    df_point_eigen_values = {
        'id': [],
        'lambda_0': [],
        'lambda_1': [],
        'lambda_2': []
    }

    # num rnn cache:
    num_rnn_cache = {}
    # heapq for non-maximum suppression:
    pq = []
    for idx_center, center in enumerate(
        points
    ):
        # find radius nearest neighbors:
        [k, idx_neighbors, _] = search_tree.search_radius_vector_3d(center, radius)

        # for each point get its nearest neighbors count:
        w = []
        deviation = []
        for idx_neighbor in np.asarray(idx_neighbors[1:]):
            # check cache:
            if not idx_neighbor in num_rnn_cache:
                [k_, _, _] = search_tree.search_radius_vector_3d(
                    points[idx_neighbor], 
                    radius
                )
                num_rnn_cache[idx_neighbor] = k_
            # update:
            w.append(num_rnn_cache[idx_neighbor])
            deviation.append(points[idx_neighbor] - center)
        
        # calculate covariance matrix:
        w = np.asarray(w)
        deviation = np.asarray(deviation)

        cov = (1.0 / w.sum()) * np.dot(
            deviation.T,
            np.dot(np.diag(w), deviation)
        )

        # get eigenvalues:
        w, _ = np.linalg.eig(cov)
        w = w[w.argsort()[::-1]]

        # add to pq:
        heapq.heappush(pq, (-w[2], idx_center))

        # add to dataframe:
        df_point_eigen_values['id'].append(idx_center)
        df_point_eigen_values['lambda_0'].append(w[0])
        df_point_eigen_values['lambda_1'].append(w[1])
        df_point_eigen_values['lambda_2'].append(w[2])
    
    # non-maximum suppression:
    suppressed = set()
    while pq:
        _, idx_center = heapq.heappop(pq)
        if not idx_center in suppressed:
            # suppress its neighbors:
            [_, idx_neighbors, _] = search_tree.search_radius_vector_3d(
                points[idx_center], 
                radius
            )
            for idx_neighbor in np.asarray(idx_neighbors[1:]):
                suppressed.add(idx_neighbor)
        else:
            continue

    # format:        
    df_point_eigen_values = pd.DataFrame.from_dict(
        df_point_eigen_values
    )

    # first apply non-maximum suppression:
    df_point_eigen_values = df_point_eigen_values.loc[
        df_point_eigen_values['id'].apply(lambda id: not id in suppressed),
        df_point_eigen_values.columns
    ]

    # then apply decreasing ratio test:
    df_point_eigen_values = df_point_eigen_values.loc[
        (df_point_eigen_values['lambda_0'] > df_point_eigen_values['lambda_1']) &
        (df_point_eigen_values['lambda_1'] > df_point_eigen_values['lambda_2']),
        df_point_eigen_values.columns
    ]

    # paint background as grey:
    pcd.paint_uniform_color([0.95, 0.95, 0.95])
    # paint keypoints as red:
    np.asarray(pcd.colors)[
        df_point_eigen_values['id'].values, :
    ] = [1.0, 0.0, 0.0]

    o3d.visualization.draw_geometries([pcd])

