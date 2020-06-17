#!/opt/conda/envs/09-point-cloud-registration/bin/python

# main.py
#     1. load point cloud in modelnet40 normal format
#     2. calculate ISS keypoints
#     3. calculate FPFH or SHOT for detected keypoints
#     3. visualize the results

import os
import argparse
import progressbar

import numpy as np
import open3d as o3d

# IO utils:
import utils.io as io
import utils.visualize as visualize

# detector:
from detection.iss import detect
# descriptor:
from description.fpfh import describe
# matcher:
from association.ransac_icp import RANSACParams, CheckerParams, ransac_match, exact_match


def main(
    input_dir, radius, bins, num_evaluations
):
    """
    Run point cloud registration on Shenlan dataset
    """
    registration_results = io.read_registration_results(
        os.path.join(input_dir, 'reg_result.txt')
    )

    # init output
    df_output = io.init_output()

    for i, r in progressbar.progressbar(
        list(
            registration_results.iterrows()
        )
    ):
        # for interactive visualization:
        if i >= num_evaluations:
            exit(0)
        
        # parse point cloud index:
        idx_target = int(r['idx1'])
        idx_source = int(r['idx2'])

        # load point clouds:
        pcd_source = io.read_point_cloud_bin(
            os.path.join(input_dir, 'point_clouds', f'{idx_source}.bin')
        )
        pcd_source, idx_inliers = pcd_source.remove_radius_outlier(nb_points=4, radius=radius)
        search_tree_source = o3d.geometry.KDTreeFlann(pcd_source)

        pcd_target = io.read_point_cloud_bin(
            os.path.join(input_dir, 'point_clouds', f'{idx_target}.bin')
        )
        pcd_target, idx_inliers = pcd_target.remove_radius_outlier(nb_points=4, radius=radius)
        search_tree_target = o3d.geometry.KDTreeFlann(pcd_target)

        # detect keypoints:
        keypoints_source = detect(pcd_source, search_tree_source, radius)
        keypoints_target = detect(pcd_target, search_tree_target, radius)

        # create descriptions:
        pcd_source_keypoints = pcd_source.select_by_index(keypoints_source['id'].values)
        fpfh_source_keypoints = o3d.registration.compute_fpfh_feature(
            pcd_source_keypoints, 
            o3d.geometry.KDTreeSearchParamHybrid(radius=5*radius, max_nn=100)
        ).data

        pcd_target_keypoints = pcd_target.select_by_index(keypoints_target['id'].values)
        fpfh_target_keypoints = o3d.registration.compute_fpfh_feature(
            pcd_target_keypoints, 
            o3d.geometry.KDTreeSearchParamHybrid(radius=5*radius, max_nn=100)
        ).data

        # generate matches:
        distance_threshold_init = 1.5 * radius
        distance_threshold_final = 1.0 * radius

        # RANSAC for initial estimation:
        init_result = ransac_match(
            pcd_source_keypoints, pcd_target_keypoints, 
            fpfh_source_keypoints, fpfh_target_keypoints,    
            ransac_params = RANSACParams(
                max_workers=5,
                num_samples=4, 
                max_correspondence_distance=distance_threshold_init,
                max_iteration=200000, 
                max_validation=500,
                max_refinement=30
            ),
            checker_params = CheckerParams(
                max_correspondence_distance=distance_threshold_init,
                max_edge_length_ratio=0.9,
                normal_angle_threshold=None
            )      
        )

        # exact ICP for refined estimation:
        final_result = exact_match(
            pcd_source, pcd_target, search_tree_target,
            init_result.transformation,
            distance_threshold_final, 60
        )

        # visualize:
        visualize.show_registration_result(
            pcd_source_keypoints, pcd_target_keypoints, init_result.correspondence_set,
            pcd_source, pcd_target, final_result.transformation
        )

        # add result:
        io.add_to_output(df_output, idx_target, idx_source, final_result.transformation)

    # write output:
    io.write_output(
        os.path.join(input_dir, 'reg_result_yaogefad.txt'),
        df_output
    )

def get_arguments():
    """ 
    Get command-line arguments

    """
    # init parser:
    parser = argparse.ArgumentParser("Point cloud registration.")

    # add required and optional groups:
    required = parser.add_argument_group('Required')
    optional = parser.add_argument_group('Optional')

    # add required:
    required.add_argument(
        "-i", dest="input", help="Input path of point cloud registration dataset.",
        required=True
    )
    required.add_argument(
        "-r", dest="radius", help="Radius for nearest neighbor search.",
        required=True, type=float
    )
    required.add_argument(
        "-b", dest="bins", help="Number of feature descriptor bins.",
        required=True, type=int
    )

    # add optional:
    optional.add_argument(
        '-n', dest='num_evaluations', help="Number of pairs to be processed for interactive estimation-evaluation.",
        required=False, type=int, default=3
    )

    # parse arguments:
    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments:
    arguments = get_arguments()

    # load registration results:
    main(
        arguments.input,
        arguments.radius,
        arguments.bins,
        arguments.num_evaluations
    )