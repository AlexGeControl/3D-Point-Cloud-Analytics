#!/opt/conda/envs/point-cloud/bin/python

# 实现voxel滤波，并加载数据集中的文件进行验证

import argparse
import os
import numpy as np
import open3d as o3d 
from pyntcloud import PyntCloud


def get_voxel_grid_classifier(points, leaf_size):
    """ Get a function for 3D point -- voxel grid assignment

    Parameters:
        points(pandas.DataFrame): points in the point cloud
    """
    # get bounding box:
    (p_min, p_max) = (points.min(), points.max())
    (D_x, D_y, D_z) = (
        np.ceil((p_max['x'] - p_min['x']) / leaf_size).astype(np.int),
        np.ceil((p_max['y'] - p_min['y']) / leaf_size).astype(np.int),
        np.ceil((p_max['z'] - p_min['z']) / leaf_size).astype(np.int),
    )
    
    def classifier(x, y, z):
        """ assign given 3D point to voxel grid

        Parameters:
            x(float): X
            y(float): Y
            z(float): Z
        
        Return:
            idx(int): voxel grid index
        """
        (i_x, i_y, i_z) = (
            np.floor((x - p_min['x']) / leaf_size).astype(np.int),
            np.floor((y - p_min['y']) / leaf_size).astype(np.int),            
            np.floor((z - p_min['z']) / leaf_size).astype(np.int),
        )

        idx = i_x + D_x * i_y + D_x * D_y * i_z

        return idx

    return classifier

def voxel_filter(points, leaf_size, method='centroid'):
    """ Downsample point cloud using voxel grid

    Parameters:
        points(pandas.DataFrame): points in the point cloud
        leaf_size(float): voxel grid resolution
        method(str): downsample method. 'centroid' or 'random'. defaults to 'centroid'

    Returns:
        filtered_points(numpy.ndarray): downsampled point cloud
    """
    filtered_points = None
    
    # TODO 03: voxel grid filtering
    working_points = points.copy(deep = True)

    # get voxel grid classifier:
    classifier = get_voxel_grid_classifier(working_points, leaf_size)
    # assign to voxel grid:
    working_points['voxel_grid_id'] = working_points.apply(
        lambda row: classifier(row['x'], row['y'], row['z']), axis = 1
    )
    
    # centroid:
    if method == 'centroid':
        filtered_points = working_points.groupby(['voxel_grid_id']).mean().to_numpy()
    elif method == 'random':
        filtered_points = working_points.groupby(['voxel_grid_id']).apply(
            lambda x: x[['x', 'y', 'z']].sample(1)
        ).to_numpy()

    return filtered_points

def main(point_cloud_filename):
    # load point cloud:
    point_cloud_pynt = PyntCloud.from_file(point_cloud_filename)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)

    # 调用voxel滤波函数，实现滤波
    filtered_cloud = voxel_filter(point_cloud_pynt.points, 10.0, method='random')
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])


def get_arguments():
    """ Get command-line arguments
    """
    # init parser:
    parser = argparse.ArgumentParser("Downsample given point cloud using voxel grid.")

    # add required and optional groups:
    required = parser.add_argument_group('Required')

    # add required:
    required.add_argument(
        "-i", dest="input", help="Input path of point cloud in ply format.",
        required=True
    )

    # parse arguments:
    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments:
    arguments = get_arguments()

    main(arguments.input)
