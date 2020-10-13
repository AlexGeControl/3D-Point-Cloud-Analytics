import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation as R

def read_point_cloud_bin(bin_path):
    """
    Read point cloud in bin format

    Parameters
    ----------
    bin_path: str
        Input path of Oxford point cloud bin

    Returns
    ----------

    """
    data = np.fromfile(bin_path, dtype=np.float32)

    # format:
    N, D = data.shape[0]// 6, 6
    point_cloud_with_normal = np.reshape(data, (N, D))

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_with_normal[:, 0:3])
    point_cloud.normals = o3d.utility.Vector3dVector(point_cloud_with_normal[:, 3:6])

    return point_cloud

def read_registration_results(results_path):
    """
    Read 

    Parameters
    ----------
    results_path: str
        Input path of point cloud registration results

    Returns
    ----------

    """
    # load csv:
    df_results = pd.read_csv(
        results_path
    )

    return df_results

def init_output():
    """
    Get registration result output template

    """
    df_output = {
        'idx1': [],
        'idx2': [],
        't_x': [],
        't_y': [],
        't_z': [],
        'q_w': [],
        'q_x': [],
        'q_y': [],
        'q_z': []
    }

    return df_output

def add_to_output(df_output, idx1, idx2, T):
    """
    Add record to output

    """
    def format_transform_matrix(T):
        r = R.from_matrix(T[:3, :3])
        q = r.as_quat()
        t = T[:3, 3]

        return (t, q)

    df_output['idx1'].append(idx1)
    df_output['idx2'].append(idx2)
    
    (t, q) = format_transform_matrix(T)

    # translation:
    df_output['t_x'].append(t[0])
    df_output['t_y'].append(t[1])
    df_output['t_z'].append(t[2])
    # rotation:
    df_output['q_w'].append(q[3])
    df_output['q_x'].append(q[0])
    df_output['q_y'].append(q[1])
    df_output['q_z'].append(q[2])

def write_output(filename, df_output):
    """
    Write output

    """
    df_output = pd.DataFrame.from_dict(
        df_output
    )

    print(f'write output to {filename}')
    df_output[
        [
            'idx1', 'idx2',
            't_x', 't_y', 't_z',
            'q_w', 'q_x', 'q_y', 'q_z'
        ]
    ].to_csv(filename, index=False)