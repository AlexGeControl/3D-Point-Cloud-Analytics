#!/opt/conda/envs/01-introduction/bin/python

# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import argparse
import os
import numpy as np
import open3d as o3d 
from pyntcloud import PyntCloud


def PCA(data, correlation=False, sort=True):
    """ Calculate PCA for given point cloud

    Parameters
    ----------
        data：点云，NX3的矩阵
        correlation：区分np的cov和corrcoef，不输入时默认为False
        sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
    
    Returns
    ----------
        eigenvalues：特征值
        eigenvectors：特征向量
    """

    # TODO 01: PCA
    # format as numpy array:
    N = data.shape[0]
    X = data.to_numpy()

    # normalize by center:
    mu = np.mean(X, axis=0)
    X_normalized = X - mu

    # get function:
    func = np.cov if not correlation else np.corrcoef
    H = func(X_normalized, rowvar=False, bias=True)

    # get eigen pairs:
    eigenvalues, eigenvectors = np.linalg.eig(H)

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors

def get_pca_o3d(w, v, points):
    """ Build open3D geometry for PCA

    Parameters
    ----------
        w: eigenvalues in descending order
        v: eigenvectors in descending order
    
    Returns
    ----------
        pca_set: o3d line set for pca visualization
    """
    # calculate centroid & variation along main axis:
    centroid = points.mean()
    projs = np.dot(points.to_numpy(), v[:,0])
    scale = projs.max() - projs.min()

    points = centroid.to_numpy() + np.vstack(
        (
            np.asarray([0.0, 0.0, 0.0]),
            scale * v.T
        )
    ).tolist()
    lines = [
        [0, 1],
        [0, 2],
        [0, 3]
    ]
    # from the largest to the smallest: RGB
    colors = np.identity(3).tolist()

    # build pca line set:
    pca_o3d = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    pca_o3d.colors = o3d.utility.Vector3dVector(colors)

    return pca_o3d

def get_surface_normals(pcd, points, knn=5):
    # create search tree:
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # init:
    N = len(pcd.points)
    normals = []

    for i in range(N):
        # find knn:
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], knn)
        # get normal:
        w, v = PCA(points.iloc[idx])
        normals.append(v[:, 0])

    return np.array(normals, dtype=np.float64)

def get_surface_normals_o3d(normals, points, scale=2):
    """ Build open3D geometry for surface normals

    Parameters
    ----------
        normals(numpy.ndarray): surface normals for each point
        points(pandas.DataFrame): points in the point cloud
        scale(float): the length of each surface normal vector

    Returns
    ----------
        surface_normals_o3d: o3d line set for surface normal visualization
    """
    # total number of points:
    N = points.shape[0]

    points = np.vstack(
        (points.to_numpy(), points.to_numpy() + scale * normals)
    )
    lines = [[i, i+N] for i in range(N)]
    colors = np.zeros((N, 3)).tolist()

    # build pca line set:
    surface_normals_o3d = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    surface_normals_o3d.colors = o3d.utility.Vector3dVector(colors)

    return surface_normals_o3d

def main(point_cloud_filename):
    # load point cloud:
    point_cloud_pynt = PyntCloud.from_file(point_cloud_filename)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)

    # 从点云中获取点，只对点进行处理
    points = point_cloud_pynt.points
    print('[PCA Normal]: Total number of points:', points.shape[0])

    # 用PCA分析点云主方向
    w, v = PCA(points)
    point_cloud_vector = v[:, 2] #点云主方向对应的向量
    print('the main orientation of this pointcloud is: ', point_cloud_vector)
    # get PCA geometry:
    pca_o3d = get_pca_o3d(w, v, points)
    
    # TODO 02:
    # 循环计算每个点的法向量
    normals = get_surface_normals(point_cloud_o3d, points)
    # 此处把法向量存放在了normals中
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    # get surface normals geometry:
    surface_normals_o3d = get_surface_normals_o3d(normals, points)
    
    # visualize point clouds with PCA and surface normals:
    o3d.visualization.draw_geometries([point_cloud_o3d, pca_o3d, surface_normals_o3d])


def get_arguments():
    """ Get command-line arguments
    """

    # init parser:
    parser = argparse.ArgumentParser("Get PCA and surface normals for given point cloud.")

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
