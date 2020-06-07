import numpy as np
import pandas as pd
import open3d as o3d

def get_spfh(point_cloud, search_tree, keypoint_id, radius, B):
    """
    Describe the selected keypoint using Simplified Point Feature Histogram(SPFH) 

    Parameters
    ----------
    point_cloud: Open3D.geometry.PointCloud
        input point cloud
    search_tree: Open3D.geometry.KDTree
        point cloud search tree
    keypoint_id: ind
        keypoint index
    radius: float
        nearest neighborhood radius
    B: float
        number of bins for each dimension

    Returns
    ----------

    """    
    # points handler:
    points = np.asarray(point_cloud.points)

    # get keypoint:
    keypoint = np.asarray(point_cloud.points)[keypoint_id]

    # find radius nearest neighbors:
    [k, idx_neighbors, _] = search_tree.search_radius_vector_3d(keypoint, radius)
    # remove query point:
    idx_neighbors = idx_neighbors[1:]
    # get normalized diff:
    diff = points[idx_neighbors] - keypoint 
    diff /= np.linalg.norm(diff, ord=2, axis=1)[:,None]

    # get n1:
    n1 = np.asarray(point_cloud.normals)[keypoint_id]
    # get u:
    u = n1
    # get v:
    v = np.cross(u, diff)
    # get w:
    w = np.cross(u, v)

    # get n2:
    n2 = np.asarray(point_cloud.normals)[idx_neighbors]
    # get alpha:
    alpha = (v * n2).sum(axis=1)
    # get phi:
    phi = (u*diff).sum(axis=1)
    # get theta:
    theta = np.arctan2((w*n2).sum(axis=1), (u*n2).sum(axis=1))

    # get alpha histogram:
    alpha_histogram = np.histogram(alpha, bins=B, range=(-1.0, +1.0))[0]
    alpha_histogram = alpha_histogram / alpha_histogram.sum()
    # get phi histogram:
    phi_histogram = np.histogram(phi, bins=B, range=(-1.0, +1.0))[0]
    phi_histogram = phi_histogram / phi_histogram.sum()
    # get theta histogram:
    theta_histogram = np.histogram(theta, bins=B, range=(-np.pi, +np.pi))[0]
    theta_histogram = theta_histogram / theta_histogram.sum()

    # build signature:
    signature = np.hstack(
        (   
            # alpha:
            alpha_histogram,
            # phi:
            phi_histogram,
            # theta:
            phi_histogram
        )
    )

    return signature

def describe(point_cloud, search_tree, keypoint_id, radius, B):
    """
    Describe the selected keypoint using Fast Point Feature Histogram(FPFH)

    Parameters
    ----------
    point_cloud: Open3D.geometry.PointCloud
        input point cloud
    search_tree: Open3D.geometry.KDTree
        point cloud search tree
    keypoint_id: ind
        keypoint index
    radius: float
        nearest neighborhood radius
    B: float
        number of bins for each dimension

    Returns
    ----------

    """
    # points handler:
    points = np.asarray(point_cloud.points)

    # get keypoint:
    keypoint = np.asarray(point_cloud.points)[keypoint_id]

    # find radius nearest neighbors:
    [k, idx_neighbors, _] = search_tree.search_radius_vector_3d(keypoint, radius)

    if k <= 1:
        return None

    # remove query point:
    idx_neighbors = idx_neighbors[1:]

    # weights:
    w = 1.0 / np.linalg.norm(
        points[idx_neighbors] - keypoint, ord=2, axis=1
    )

    # SPFH from neighbors:
    X = np.asarray(
        [get_spfh(point_cloud, search_tree, i, radius, B) for i in idx_neighbors]
    )

    # neighborhood contribution:
    spfh_neighborhood = 1.0 / (k - 1) * np.dot(w, X)

    # query point spfh:
    spfh_query = get_spfh(point_cloud, search_tree, keypoint_id, radius, B)

    # finally:
    spfh = spfh_query + spfh_neighborhood

    # normalize again:
    spfh = spfh / np.linalg.norm(spfh)

    return spfh
