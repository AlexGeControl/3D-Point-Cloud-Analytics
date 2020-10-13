import numpy as np
import pandas as pd
import open3d as o3d

def get_spfh(point_cloud, search_tree, keypoint_id, search_params, B):
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
    search_params: o3d.geometry.KDTreeSearchParamHybrid
        nearest neighborhood radius and max num. of nns
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
    [k, idx_neighbors, _] = search_tree.search_hybrid_vector_3d(keypoint, search_params.radius, search_params.max_nn)

    if k <= 1:
        return None

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
    phi = (u * diff).sum(axis=1)
    # get theta:
    theta = np.arctan2((w*n2).sum(axis=1), (u*n2).sum(axis=1))

    # get alpha histogram:
    alpha_histogram = np.histogram(alpha, bins=B, range=(-1.0, +1.0), density=True)[0]
    # get phi histogram:
    phi_histogram = np.histogram(phi, bins=B, range=(-1.0, +1.0), density=True)[0]
    # get theta histogram:
    theta_histogram = np.histogram(theta, bins=B, range=(-np.pi, +np.pi), density=True)[0]

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

def describe(point_cloud, search_params, B):
    """
    Describe the selected keypoint using Fast Point Feature Histogram(FPFH)

    Parameters
    ----------
    point_cloud: Open3D.geometry.PointCloud
        input point cloud
    search_params: o3d.geometry.KDTreeSearchParamHybrid
        nearest neighborhood radius and max num. of nns
    B: float
        number of bins for each dimension

    Returns
    ----------

    """
    # points handler:
    points = np.asarray(point_cloud.points)
    N, _ = points.shape
    search_tree = o3d.geometry.KDTreeFlann(point_cloud)

    # spfh cache:
    spfh_lookup_table = {}
    # output:
    description = []

    # calculate FPFH for all the points:
    for keypoint_id in range(N):
        # get keypoint:
        keypoint = np.asarray(point_cloud.points)[keypoint_id]

        # find radius nearest neighbors:
        [k, idx_neighbors, dis_neighbors] = search_tree.search_hybrid_vector_3d(keypoint, search_params.radius, search_params.max_nn)  

        if k <= 1:
            return None      

        # remove query point:
        k = k - 1
        idx_neighbors = idx_neighbors[1:]   
        dis_neighbors = dis_neighbors[1:] 

        # FPFH from neighbors:
        # a. descriptions:
        X = []
        for j in idx_neighbors:
            spfh_neighbor = spfh_lookup_table.get(j, None)
            if spfh_neighbor is None:
                spfh_lookup_table[j] = get_spfh(point_cloud, search_tree, j, search_params, B)
            X.append(spfh_lookup_table[j])
        X = np.asarray(X)
        # b. weights:
        w = 1.0 / np.asarray(dis_neighbors)
        # w = w / w.sum()
        # c. weighed average:
        spfh_neighborhood = 1.0 / k * np.dot(w, X)

        # query point spfh:
        spfh_keypoint = spfh_lookup_table.get(keypoint_id, None)
        if spfh_keypoint is None:
            spfh_lookup_table[keypoint_id] = get_spfh(point_cloud, search_tree, j, search_params, B)
        spfh_keypoint = spfh_lookup_table[keypoint_id]

        # finally:
        fpfh_keypoint = spfh_keypoint + spfh_neighborhood

        description.append(fpfh_keypoint)

    # reshape as Open3D:
    description = np.asarray(description).T

    # normalize again:
    # description = description / np.linalg.norm(description, axis=1)

    return description
