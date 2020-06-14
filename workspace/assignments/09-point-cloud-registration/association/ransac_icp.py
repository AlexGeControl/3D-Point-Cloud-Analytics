import collections
import copy

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
import open3d as o3d

# RANSAC configuration:
RANSACParams = collections.namedtuple(
    'RANSACParams',
    ['num_samples', 'max_correspondence_distance', 'max_iteration', 'max_validation', 'max_refinement']
)

# fast pruning algorithm configuration:
CheckerParams = collections.namedtuple(
    'CheckerParams', 
    ['max_correspondence_distance', 'max_edge_length_ratio', 'normal_angle_threshold']
)

# result:
Result = collections.namedtuple(
    'Result', 
    ['num_iteration', 'num_validation', 'registration_result']
)

def get_potential_matches(feature_source, feature_target):
    """
    Get potential matches

    Parameters
    ----------
    feature_source: open3d.registration.Feature
        feature descriptions of source point cloud
    feature_target: open3d.registration.Feature
        feature descriptions of target point cloud

    Returns
    ----------
    matches: numpy.ndarray
        potential matches as N-by-2 numpy.ndarray

    """
    # build search tree on target features:
    search_tree = o3d.geometry.KDTreeFlann(feature_target.data)

    # generate nearest-neighbor match for all the points in the source:
    N = feature_source.num()
    matches = []
    for i in range(N):
        query = feature_source.data[:, i]
        _, idx_nn_target, _ = search_tree.search_knn_vector_xd(query, 1)
        matches.append(
            [i, idx_nn_target[0]]
        )

    # format:
    matches = np.asarray(
        matches
    )

    return matches

def solve_icp(P, Q):
    """
    Solve ICP

    Parameters
    ----------
    P: numpy.ndarray
        source point cloud as N-by-3 numpy.ndarray
    Q: numpy.ndarray
        target point cloud as N-by-3 numpy.ndarray

    Returns
    ----------
    T: transform matrix as 4-by-4 numpy.ndarray
        transformation matrix from one-step ICP

    """
    # compute centers:
    up = P.mean(axis = 0)
    uq = Q.mean(axis = 0)

    # move to center:
    P_centered = P - up
    Q_centered = Q - uq

    U, s, V = np.linalg.svd(np.dot(Q_centered.T, P_centered), full_matrices=True, compute_uv=True)
    R = np.dot(U, V)
    t = uq - np.dot(R, up)

    # format as transform:
    T = np.zeros((4, 4))
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    T[3, 3] = 1.0

    return T

def is_valid_match(
    pcd_source, pcd_target,
    proposal,
    checker_params 
):
    """
    Check proposal validity using the fast pruning algorithm

    Parameters
    ----------
    pcd_source: open3d.geometry.PointCloud
        source point cloud
    pcd_target: open3d.geometry.PointCloud
        target point cloud
    proposal: numpy.ndarray
        RANSAC potential as num_samples-by-2 numpy.ndarray
    checker_params:
        fast pruning algorithm configuration

    Returns
    ----------
    T: transform matrix as numpy.ndarray or None
        whether the proposal is a valid match for validation

    """
    idx_source, idx_target = proposal[:,0], proposal[:,1]

    # TODO: this checker should only be used for pure translation
    if not checker_params.normal_angle_threshold is None:
        # get corresponding normals:
        normals_source = np.asarray(pcd_source.normals)[idx_source]
        normals_target = np.asarray(pcd_target.normals)[idx_target]

        # a. normal direction check:
        normal_cos_distances = (normals_source*normals_target).sum(axis = 1)
        is_valid_normal_match = np.all(normal_cos_distances >= np.cos(checker_params.normal_angle_threshold)) 

        if not is_valid_normal_match:
            return None

    # get corresponding points:
    points_source = np.asarray(pcd_source.points)[idx_source]
    points_target = np.asarray(pcd_target.points)[idx_target]

    # b. edge length ratio check:
    pdist_source = pdist(points_source)
    pdist_target = pdist(points_target)
    is_valid_edge_length = np.all(
        np.logical_and(
            pdist_source > checker_params.max_edge_length_ratio * pdist_target,
            pdist_target > checker_params.max_edge_length_ratio * pdist_source
        )
    )

    if not is_valid_edge_length:
        return None

    # c. fast correspondence distance check:s
    T = solve_icp(points_source, points_target)
    R, t = T[0:3, 0:3], T[0:3, 3]
    deviation = np.linalg.norm(
        points_target - np.dot(points_source, R.T) - t,
        axis = 1
    )
    is_valid_correspondence_distance = np.all(deviation <= checker_params.max_correspondence_distance)

    return T if is_valid_correspondence_distance else None

def exact_match(
    pcd_source, pcd_target, search_tree_target,
    T,
    max_correspondence_distance, max_iteration
):
    """
    Perform exact match on given point cloud pair

    Parameters
    ----------
    pcd_source: open3d.geometry.PointCloud
        source point cloud
    pcd_target: open3d.geometry.PointCloud
        target point cloud
    search_tree_target: scipy.spatial.KDTree
        target point cloud search tree
    T: numpy.ndarray
        transform matrix as 4-by-4 numpy.ndarray
    max_correspondence_distance: float
        correspondence pair distance threshold
    max_iteration:
        max num. of iterations 

    Returns
    ----------
    T: numpy.ndarray
        transform matrix as 4-by-4 numpy.ndarray

    """
    # num. points in the source:
    N = len(pcd_source.points)

    for _ in range(max_iteration):
        # TODO: transform is actually an in-place operation. deep copy first otherwise the result will be WRONG
        pcd_source_current = copy.deepcopy(pcd_source)
        # apply transform:
        pcd_source_current = pcd_source_current.transform(T)
        
        # find correspondence:
        matches = []
        for n in range(N):
            query = np.asarray(pcd_source_current.points)[n]
            _, idx_nn_target, dis_nn_target = search_tree_target.search_knn_vector_3d(query, 1)

            if dis_nn_target[0] <= max_correspondence_distance:
                matches.append(
                    [n, idx_nn_target[0]]
                )
        matches = np.asarray(matches)

        if len(matches) >= 4:
            # sovle ICP:
            P = np.asarray(pcd_source.points)[matches[:,0]]
            Q = np.asarray(pcd_target.points)[matches[:,1]]
            T = solve_icp(P, Q)
    
    return T

def ransac_match(
    pcd_source, pcd_target, 
    feature_source, feature_target,
    ransac_params, checker_params
):
    """
    Perform RANSAC match on given point cloud pair

    Parameters
    ----------
    pcd_source: open3d.geometry.PointCloud
        source point cloud
    pcd_target: open3d.geometry.PointCloud
        target point cloud
    ransac_params:
        RANSAC configuration
    checker_params:
        fast pruning algorithm configuration

    Returns
    ----------
    best_result: Result
        best matching result from RANSAC ICP

    """
    # identify potential matches:
    matches = get_potential_matches(feature_source, feature_target)

    # build search tree on the target:
    search_tree_target = o3d.geometry.KDTreeFlann(pcd_target)

    # RANSAC:
    N, _ = matches.shape
    idx_matches = np.arange(N)

    num_validation = 0
    
    T = None 
    best_result = None

    # get at least one valid proposal:
    while T is None:
        # generate proposal:
        proposal = matches[
            np.random.choice(idx_matches, ransac_params.num_samples, replace=False)
        ]
        # check validity:
        T = is_valid_match(
            pcd_source, pcd_target,
            proposal,
            checker_params      
        )
    print('[RANSAC ICP]: Get valid proposal. Start registration...')

    # RANSAC:
    for i in range(ransac_params.max_iteration):
        # check validity:
        if (not (T is None)) and (num_validation < ransac_params.max_validation):
            num_validation += 1

            # refine estimation on all keypoints:
            T = exact_match(
                pcd_source, pcd_target, search_tree_target,
                T,
                ransac_params.max_correspondence_distance, 
                ransac_params.max_refinement
            )

            result = Result(
                num_iteration=(i+1), 
                num_validation=num_validation,
                registration_result=o3d.registration.evaluate_registration(
                    pcd_source, pcd_target, ransac_params.max_correspondence_distance, T
                )
            )

            if best_result is None:
                best_result = result
            
            # update best result:
            best_result = best_result if best_result.registration_result.fitness > result.registration_result.fitness else result

            if num_validation == ransac_params.max_validation:
                break

        # generate proposal:
        proposal = matches[
            np.random.choice(idx_matches, ransac_params.num_samples, replace=False)
        ]

        T = is_valid_match(
            pcd_source, pcd_target,
            proposal,
            checker_params      
        )

    return best_result

