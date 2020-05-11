#!/opt/conda/envs/point-cloud/bin/python

# Copyright (C) Ge Yao, alexgecontrol@qq.com.
#
# Point cloud filters based on Python PCL
#
# All Rights Reserved.

# Author: Ge Yao

class VoxelFilter():
    """
    Voxel filter

    Parameters
    ----------
    cloud: pcl.cloud
        PCL point cloud
    leaf_size: float
        voxel dimension

    Attributes
    ----------

    """
    def __init__(self, cloud, leaf_size = 0.0618):
        self.__leaf_size = leaf_size

        self.__filter = cloud.make_voxel_grid_filter()
        self.__filter.set_leaf_size(
            *([self.__leaf_size]*3)
        )

    def filter(self):
        """
        Generate filtered point cloud
        """
        return self.__filter.filter()

class ROIFilter():
    """
    ROI filter

    Parameters
    ----------
    cloud: pcl.cloud
        PCL point cloud
    name: string
        filter field name
    limits: tuple
        filter field value range, in (min, max) format

    Attributes
    ----------

    """
    def __init__(
        self,
        cloud,
        name,
        limits
    ):
        self.__name = name
        self.__limits = limits

        self.__filter = cloud.make_passthrough_filter()
        self.__filter.set_filter_field_name(
            self.__name
        )
        self.__filter.set_filter_limits(
            *self.__limits
        )

    def filter(self):
        """
        Generate filtered point cloud
        """
        return self.__filter.filter()

class OutlierFilter():
    """
    Outlier filter

    Parameters
    ----------
    cloud: pcl.cloud
        PCL point cloud
    k: int
        the number of nearest neighbor points to analyze for any given point
    factor: float
        maximum deviation factor. Any point with a mean_distance larger than (global_mean_distance + factor*global_std_dev) will be considered outlier

    Attributes
    ----------

    """
    def __init__(self, cloud, k = 50, factor = 1):
        self.__k = k
        self.__factor = factor

        self.__filter = cloud.make_statistical_outlier_filter()
        self.__filter.set_mean_k(self._k)
        self.__filter.set_std_dev_mul_thresh(self._factor)

    def filter(self):
        """
        Generate filtered point cloud
        """
        return self.__filter.filter()