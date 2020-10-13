import numpy as np
import open3d as o3d 


def show_inlier_outlier(cloud, ind):
    """
    Visualize inliers and outliers

    """
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def get_point_cloud_diameter(pcd):
    """
    Get point cloud diameter by min-max bounding box

    """
    diameter = np.linalg.norm(
        pcd.get_max_bound() - pcd.get_min_bound()
    )

    return diameter

def show_registration_result(
    pcd_source_keypoints, pcd_target_keypoints, association,
    pcd_source_dense, pcd_target_dense, transformation
):
    """
    Visualize point cloud registration results.

    Parameters
    ----------
    pcd_source_keypoints: open3d.geometry.PointCloud
        keypoints in source point cloud
    pcd_target_keypoints: open3d.geometry.PointCloud
        keypoints in target point cloud
    association: numpy.ndarray
        keypoint associations from feature matching
    pcd_source_dense: open3d.geometry.PointCloud
        filtered source point cloud
    pcd_target_dense: open3d.geometry.PointCloud
        filtered target point cloud
    transformation: numpy.ndarray
        transformation matrix

    Returns
    ----------
    None

    """
    #
    # group 01 -- registration result:
    # 

    # apply transform:    
    pcd_source_dense.transform(transformation)

    # move registration result to origin:
    center_source, _ = pcd_source_dense.compute_mean_and_covariance()
    center_target, _ = pcd_target_dense.compute_mean_and_covariance()

    translation = 0.5 * (center_source + center_target)

    pcd_source_dense_centered = pcd_source_dense.translate(-translation)
    pcd_target_dense_centered = pcd_target_dense.translate(-translation)

    # draw result:
    pcd_source_dense_centered.paint_uniform_color([1, 0.706, 0])
    pcd_target_dense_centered.paint_uniform_color([0, 0.651, 0.929])

    #
    # group 02 -- keypoint association result:
    # 

    # get diameters of source and target:
    diameter_source = get_point_cloud_diameter(pcd_source_dense)
    diameter_target = get_point_cloud_diameter(pcd_target_dense)

    # shift correspondence sets:
    diameter = max(diameter_source, diameter_target)

    pcd_source_keypoints_shifted = pcd_source_keypoints.translate(
        -translation + np.asarray([diameter, -diameter, 0.0])
    )
    pcd_target_keypoints_shifted = pcd_target_keypoints.translate(
        -translation + np.asarray([diameter, +diameter, 0.0])
    )

    # draw associations:
    association = np.asarray(association)[:20,:]
    N, _ = association.shape
    # keep only 20 pairs:
    points = np.vstack(
        (
            np.asarray(pcd_source_keypoints_shifted.points)[association[:, 0]],
            np.asarray(pcd_target_keypoints_shifted.points)[association[:, 1]]
        )
    )
    correspondences = np.asarray(
        [
            [i, i + N] for i in np.arange(N)
        ]
    )
    colors = [
        [0.0, 0.0, 0.0] for i in range(N)
    ]
    correspondence_set = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(points),
        lines = o3d.utility.Vector2iVector(correspondences),
    )
    correspondence_set.colors = o3d.utility.Vector3dVector(colors)

    pcd_source_keypoints_shifted.paint_uniform_color([0.0, 1.0, 0.0])
    np.asarray(pcd_source_keypoints_shifted.colors)[association[:, 0], :] = [1.0, 0.0, 0.0]
    pcd_target_keypoints_shifted.paint_uniform_color([0.0, 0.0, 1.0])
    np.asarray(pcd_target_keypoints_shifted.colors)[association[:, 1], :] = [1.0, 0.0, 0.0]
    o3d.visualization.draw_geometries(
        [   
            # registration result:
            pcd_source_dense_centered, pcd_target_dense_centered, 
            # feature point association:
            pcd_source_keypoints_shifted, pcd_target_keypoints_shifted,correspondence_set
        ]
    )
