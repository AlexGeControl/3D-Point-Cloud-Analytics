import pandas as pd
import open3d as o3d

def read_modelnet40_normal(filepath):
    '''
    Read ModelNet40 sample with normal as Open3D point cloud

    Parameters
    ----------
    filepath: str
        File path of ModelNet40 sample

    Returns
    ----------
    point_cloud: Open3D.geometry.PointCloud
        Open3D point cloud

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
    
    point_cloud = o3d.geometry.PointCloud()

    point_cloud.points = o3d.utility.Vector3dVector(
        df_point_cloud_with_normal[['x', 'y', 'z']].values
    )
    point_cloud.normals = o3d.utility.Vector3dVector(
        df_point_cloud_with_normal[['nx', 'ny', 'nz']].values
    )

    return point_cloud