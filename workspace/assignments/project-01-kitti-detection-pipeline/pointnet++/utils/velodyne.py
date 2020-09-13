import struct
import numpy as np


def read_measurements(filepath):
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

def transform_to_cam(X_velo, param):
    '''
    Transform point from Velodyne frame to camera frame

    Parameters
    ----------
    X_velo: numpy.ndarray
        points in velodyne frame
    param: dict
        Vehicle parameters

    Returns
    ----------
    X_cam: numpy.ndarray
        Points in camera frame

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

    return X_cam

def transform_to_pixel(X_cam, param):
    '''
    Transform point from camera frame to pixel frame

    Parameters
    ----------
    X_cam: numpy.ndarray
        points in camera frame
    param: dict
        Vehicle parameters

    Returns
    ----------
    X_pixel: numpy.ndarray
        Points in pixel frame

    '''
    # get params:
    K, b = param['P2'][:,0:3], param['P2'][:,3]

    # project to pixel frame:
    X_pixel = np.dot(
        K, X_cam.T
    ).T + b

    # rectify:
    X_pixel = (X_pixel[:, :2].T / X_pixel[:, 2]).T

    return X_pixel