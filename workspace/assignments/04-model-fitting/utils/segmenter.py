#!/opt/conda/envs/point-cloud/bin/python

# Copyright (C) Ge Yao, alexgecontrol@qq.com.
#
# Point cloud filters based on Python PCL
#
# All Rights Reserved.

# Author: Ge Yao

import pcl
import numpy as np

class GroundSegmenter():
    """
    Ground segmenter

    Parameters
    ----------
    cloud: pcl.cloud
        PCL point cloud
    max_distance: float
        RANSAC inlier threshold tau

    Attributes
    ----------

    """
    def __init__(self, cloud, max_distance = 0.30):
        self.__max_distance = max_distance

        self.__segmenter = cloud.make_segmenter()
        self.__segmenter.set_model_type(pcl.SACMODEL_PLANE)
        self.__segmenter.set_method_type(pcl.SAC_RANSAC)
        self.__segmenter.set_distance_threshold(self.__max_distance)

    def segment(self):
        """
        Get segmented ground plane
        """
        return self.__segmenter.segment()

    def get_max_distance(self):
        """
        Get plane RANSAC threshold tau
        """
        return self.__max_distance

class DBSCANSegmenter():
    """
    DBSCAN segmenter for surrounding objects

    Parameters
    ----------
    cloud: pcl.cloud
        PCL point cloud
    eps: float
        RANSAC inlier threshold tau
    min_samples: int
        RANSAC inlier threshold tau
    max_samples: int
        RANSAC inlier threshold tau

    Attributes
    ----------

    """
    def __init__(
        self,
        cloud,
        eps = 0.001, min_samples = 10, max_samples = 250
    ):
        """ Instantiate Euclidean segmenter
        """
        # 1. Convert XYZRGB to XYZ:
        self.__cloud = cloud
        self.__tree = self.__cloud.make_kdtree()

        # 2. Set params:
        self.__eps = eps
        self.__min_samples = min_samples
        self.__max_samples = max_samples

        # 3. Create segmenter:
        self.__segmenter = self.__cloud.make_EuclideanClusterExtraction()
        self.__segmenter.set_ClusterTolerance(self.__eps)
        self.__segmenter.set_MinClusterSize(self.__min_samples)
        self.__segmenter.set_MaxClusterSize(self.__max_samples)
        self.__segmenter.set_SearchMethod(self.__tree)

    def segment(self):
        """ Segment objects
        """
        # 1. Segment objects:
        cluster_indices = self.__segmenter.Extract()

        # 2. Generate representative point for object:
        cluster_reps = []
        for idx_points in cluster_indices:
            object_cloud = self.__cloud.extract(idx_points)

            # Use centroid as representative point:
            rep_position = np.mean(
                object_cloud.to_array(),
                axis=0
            )[:3]

            cluster_reps.append(rep_position)

        return (cluster_indices, cluster_reps)