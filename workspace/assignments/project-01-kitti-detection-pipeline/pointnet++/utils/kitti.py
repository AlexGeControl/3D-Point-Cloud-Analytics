import numpy as np
import open3d as o3d
import pandas as pd

from .velodyne import transform_to_cam, transform_to_pixel

def get_orientation_in_camera_frame(X_cam_centered):
    """
    Get object orientation using PCA
    """
    # keep only x-z:
    X_cam_centered = X_cam_centered[:, [0, 2]]

    H = np.cov(X_cam_centered, rowvar=False, bias=True)

    # get eigen pairs:
    eigenvalues, eigenvectors = np.linalg.eig(H)

    idx_sort = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx_sort]
    eigenvectors = eigenvectors[:, idx_sort]

    # orientation as arctan2(-z, x):
    return np.arctan2(-eigenvectors[0][1], eigenvectors[0][0])

def to_kitti_eval_format(segmented_objects, object_ids, param, predictions, decoder):
    """
    Write prediction result as KITTI evaluation format

    Parameters
    ----------
    segmented_objects: open3d.geometry.PointCloud
        Point cloud of segmented objects
    object_ids: numpy.ndarray
        Object IDs as numpy.ndarray
    predictions:
        Object Predictions

    Returns
    ----------

    """
    # parse params:
    points = np.asarray(segmented_objects.points)

    # initialize KITTI label:
    label = {
        'type': [],
        'left': [], 'top': [], 'right': [], 'bottom': [],
        'height': [], 'width': [], 'length': [],
        'cx': [], 'cy': [], 'cz': [], 
        'ry': [], 
        # between 0 and 100:
        'score': []
    }
    formatter = lambda x: f'{x:.2f}'
    kitti_type = {
        'vehicle': 'Car',
        'pedestrian': 'Pedestrian',
        'cyclist': 'Cyclist',
        'misc': 'Misc'
    }

    for class_id in predictions:
        # get color
        class_name = decoder[class_id]

        if (class_name == 'misc'):
            continue
        
        # get KITTI type:
        class_name = kitti_type[class_name]

        # show instances:
        for object_id in predictions[class_id]:
            # set object type:
            label['type'].append(class_name)

            # transform to camera frame:
            X_velo = points[object_ids == object_id]
            X_cam = transform_to_cam(X_velo, param)

            # transform to pixel frame:
            X_pixel = transform_to_pixel(X_cam, param)

            # set 2D bounding box:
            top_left = X_pixel.min(axis = 0)
            bottom_right = X_pixel.max(axis = 0)

            label['left'].append(formatter(top_left[0]))
            label['top'].append(formatter(top_left[1]))
            label['right'].append(formatter(bottom_right[0]))
            label['bottom'].append(formatter(bottom_right[1]))

            # set object location:
            c_center = X_cam.mean(axis = 0)

            label['cx'].append(formatter(c_center[0]))
            label['cy'].append(formatter(c_center[1]))
            label['cz'].append(formatter(c_center[2]))

            # set object orientation:
            X_cam_centered = X_cam - c_center
            orientation = get_orientation_in_camera_frame(X_cam_centered)
            label['ry'].append(formatter(orientation))

            # project to object frame:
            cos_ry = np.cos(-orientation)
            sin_ry = np.sin(-orientation)

            R_obj_to_cam = np.asarray(
                [
                    [ cos_ry, 0.0, sin_ry],
                    [    0.0, 1.0,    0.0],
                    [-sin_ry, 0.0, cos_ry]
                ]
            )

            X_obj = np.dot(
                R_obj_to_cam.T, (X_cam_centered).T
            ).T

            # set object dimension:
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(
                X_obj
            )
            bounding_box = object_pcd.get_axis_aligned_bounding_box()
            extent = bounding_box.get_extent()

            # height along y-axis:
            label['height'].append(formatter(extent[1]))
            # width along x-axis:
            label['width'].append(formatter(extent[0]))
            # length along z-axis:
            label['length'].append(formatter(extent[2]))

            # set confidence:
            confidence = 100.0 * predictions[class_id][object_id]
            label['score'].append(formatter(confidence))
    
    # format as pandas dataframe:
    label = pd.DataFrame.from_dict(
        label
    )
    
    # set value for unavailable fields:
    label['truncated'] = -1
    label['occluded'] = -1
    # don't evaluate AOS:
    label['alpha'] = -10

    # set column order:
    label = label[
        [
            'type',
            'truncated',
            'occluded',
            'alpha',
            'left', 'top', 'right', 'bottom',
            'height', 'width', 'length',
            'cx', 'cy', 'cz', 'ry',
            'score'
        ]
    ]

    return label