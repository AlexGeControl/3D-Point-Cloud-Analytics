import heapq
import numpy as np
import pandas as pd
import open3d as o3d

def detect(point_cloud, search_tree, radius):
    """
    Detect point cloud key points using Intrinsic Shape Signature(ISS)

    Parameters
    ----------
    point_cloud: Open3D.geometry.PointCloud
        input point cloud
    search_tree: Open3D.geometry.KDTree
        point cloud search tree
    radius: float
        radius for ISS computing

    Returns
    ----------
    point_cloud: numpy.ndarray
        Velodyne measurements as N-by-3 numpy ndarray

    """
    # points handler:
    points = np.asarray(point_cloud.points)

    # keypoints container:
    keypoints = {
        'id': [],
        'x': [],
        'y': [],
        'z': [],
        'lambda_0': [],
        'lambda_1': [],
        'lambda_2': []
    }

    # cache for number of radius nearest neighbors:
    num_rnn_cache = {}
    # heapq for non-maximum suppression:
    pq = []
    for idx_center, center in enumerate(
        points
    ):
        # find radius nearest neighbors:
        [k, idx_neighbors, _] = search_tree.search_radius_vector_3d(center, radius)

        # for each point get its nearest neighbors count:
        w = []
        deviation = []
        for idx_neighbor in np.asarray(idx_neighbors[1:]):
            # check cache:
            if not idx_neighbor in num_rnn_cache:
                [k_, _, _] = search_tree.search_radius_vector_3d(
                    points[idx_neighbor], 
                    radius
                )
                num_rnn_cache[idx_neighbor] = k_
            # update:
            w.append(num_rnn_cache[idx_neighbor])
            deviation.append(points[idx_neighbor] - center)
        
        # calculate covariance matrix:
        w = np.asarray(w)
        deviation = np.asarray(deviation)

        cov = (1.0 / w.sum()) * np.dot(
            deviation.T,
            np.dot(np.diag(w), deviation)
        )

        # get eigenvalues:
        w, _ = np.linalg.eig(cov)
        w = w[w.argsort()[::-1]]

        # add to pq:
        heapq.heappush(pq, (-w[2], idx_center))

        # add to dataframe:
        keypoints['id'].append(idx_center)
        keypoints['x'].append(center[0])
        keypoints['y'].append(center[1])
        keypoints['z'].append(center[2])
        keypoints['lambda_0'].append(w[0])
        keypoints['lambda_1'].append(w[1])
        keypoints['lambda_2'].append(w[2])
    
    # non-maximum suppression:
    suppressed = set()
    while pq:
        _, idx_center = heapq.heappop(pq)
        if not idx_center in suppressed:
            # suppress its neighbors:
            [_, idx_neighbors, _] = search_tree.search_radius_vector_3d(
                points[idx_center], 
                radius
            )
            for idx_neighbor in np.asarray(idx_neighbors[1:]):
                suppressed.add(idx_neighbor)
        else:
            continue

    # format:        
    keypoints = pd.DataFrame.from_dict(
        keypoints
    )

    # first apply non-maximum suppression:
    keypoints = keypoints.loc[
        keypoints['id'].apply(lambda id: not id in suppressed),
        keypoints.columns
    ]

    # then apply decreasing ratio test:
    keypoints = keypoints.loc[
        (keypoints['lambda_0'] > keypoints['lambda_1']) &
        (keypoints['lambda_1'] > keypoints['lambda_2']),
        keypoints.columns
    ]

    return keypoints