#!/bin/bash

# download left color images:
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip

# download calibration results:
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip

# download labels:
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip

# download Velodyne point clouds:
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip

# download development kit:
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip