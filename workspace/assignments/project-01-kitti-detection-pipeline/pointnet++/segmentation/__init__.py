import numpy as np
import open3d as o3d

from collections import namedtuple

SurroundingObject = namedtuple(
    'SurroundingObject',
    ['point_cloud', 'bounding_box']
)

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
