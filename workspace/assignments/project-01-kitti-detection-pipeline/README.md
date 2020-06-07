# Detection Pipeline on KITTI 3D Object Dataset

Python object detection pipeline on KITTI 3D Object dataset for Capstone Project One of [3D Point Cloud Processing](https://www.shenlanxueyuan.com/course/204) from [深蓝学院](https://www.shenlanxueyuan.com/)

---

## Solution Guide

---

### Build Object Classification Dataset from KITTI 3D Object

Before deep learning model training, a high quality object classification dataset must be built out of the original KITTI 3D Object dataset. The ETL tool, implemented in Python, is available at (click to follow the link) **[/workspace/assignments/project-01-kitti-detection-pipeline/extract.py](extract.py)**

#### Algorithm Workflow

The algorithm workflow in pseudo code is as follows. For each frame(velodyne-image-calib-label tuple) from KITTI 3D Object:

* First, perform **ground plane & object segmentation** on **Velodyne measurements**.
    * The implementation from [/workspace/assignments/04-model-fitting/clustering.py](assignment 04 of model fitting) is used here for region proposal 
* Build **radius-nearest neighbor** search tree upon segmented objects for later **label association**
* Then load label and associate its info with segmented objects as follows:
    * First, map **object center** from **camera frame** to **velodyne frame** using the parameters from corresponding calib file.
    * Query the search tree and identify a **coarse ROI**. 
        * The radius is determined using the dimensions of the bounding box. Euclidean transform is isometry.
    * Map the point measurements within the sphere into **object frame**.
    * Identify a **refined ROI** through **object bounding box filtering**.
    * Perform **non-maximum suppression on segmented object ids** for final label association:
        * Perform an ID-count on points inside the bounding box.
        * Identify the segmented object with maximum number of measurements inside the bounding box.
        * Associate the KITTI 3D Object label with the segmented object of maximum ID-count.
* Finally, write the **point cloud with normal** and corresponding **metadata** info persistent storage. 

#### Usage

Run the following commands to use the ETL tool:

```bash
# go to workspace:
cd /workspace/assignments/project-01-kitti-detection-pipeline
# activate environment:
conda activate kitti-detection-pipeline
# perform ETL on KITTI 3D Object dataset:
./extract.py -i /workspace/data/kitti-3d-object-detection/training/ -o /workspace/data/kitti_3d_object_classification_normal/
```

#### Visualization of Extracted Dataset

Sample visualization of the extracted classification dataset is as follows:

Camera View                |Lidar Pipeline View
:-------------------------:|:-------------------------:
![Sample 01 Camera View](doc/dataset-sample-01-camera-view.png)  |  ![Sample 01 Lidar Pipeline View](doc/dataset-sample-01-lidar-pipeline-view.png)
![Sample 02 Camera View](doc/dataset-sample-02-camera-view.png)  |  ![Sample 02 Lidar Pipeline View](doc/dataset-sample-02-lidar-pipeline-view.png)
![Sample 03 Camera View](doc/dataset-sample-03-camera-view.png)  |  ![Sample 03 Lidar Pipeline View](doc/dataset-sample-03-lidar-pipeline-view.png)

--- 

### Dataset Analytics before Deep Learning Modelling

Before training the network, data quality check must be performed to minimize the effect of uneven class distribution, etc, on model building.

#### Class Distribution

![Dataset Analytics, Class Distribution](doc/dataset-analysis-class-distribution.png)

The above visualization shows that the dataset has a **significantly uneven class distribution**. So measures must be taken to mitigate its effect on network training:

* Perform data augmentation through random rotation along z-axis, etc, to introduct more training instances
* Use focal loss for model optimization

The class distribution after resampling is show below. After resampling the dataset is more evenly distributed.

![Dataset Analytics, Resampled Class Distribution](doc/resampling-class-distribution.png)

#### Influence of Distance on Measurement Density

For efficient deep-learning network training, all input point clouds should be transformed to the same size. However, the number and density of lidar measurements is influenced by the object's distance to ego vehicle. So measurement count analytics must be performed before choosing FoV and input size.

![Dataset Analytics, Measurement Count by Object Distance](doc/dataset-analysis-measurement-count-distance.png)

From the above visualization it's obvious that the number of measurements will drop significantly as the object moves away from ego vehicle. Since for real autonomous driving system only object which lies inside local map matters, here objects which are too far away from ego vehicle will be filtered out from the training set. The threshold can be determined using the local visualization below

![Dataset Analytics, Measurement Count by Object Distance for ROI Selection](doc/dataset-analysis-measurement-count-distance-roi-selection.png)

Sample visualization of the resampled object point clouds are as follows. The ROI is defined as the area **<= 25 meters** away from ego vehicle. The measurements of each point cloud is resampled to **64** points for efficient deep network processing. 

Side View                |Top Down View
:-------------------------:|:-------------------------:
![Side View Vehicle](doc/resampling-vehicle-view-01.png)  |  ![Top Down View Vehicle](doc/resampling-vehicle-view-02.png)
![Side View Pedestrian](doc/resampling-pedestrian-view-01.png)  |  ![Top Down View Pedestrian](doc/resampling-pedestrian-view-02.png)
![Side View Cyclist](doc/resampling-cyclist-view-01.png)  |  ![Top Down View Cyclist](doc/resampling-cyclist-view-02.png)

---

### Object Classification Network

PointNet++ with focal loss is used as object classification network.

#### Up & Running

Run the following commands to train the network on the resampled dataset:

```bash
# go to workspace:
cd /workspace/assignments/project-01-kitti-detection-pipeline/pointnet++
# activate environment:
conda activate kitti-detection-pipeline
# compile pointnet++ laysers:
tf_ops/compile_ops.sh
# train the network on the resampled dataset:
./train_modelnet.py
# visualize training loss on Tensorboard:
tensorboard --logdir=logs --bind_all --port=6006
```

After this the training loss can be monitored inside local browser at http://localhost:46006/

#### Training

The loss and categorical accuracy as training goes are shown below:

<img src="doc/training-loss.png" alt="Training-Loss">

<img src="doc/training-accuracy.png" alt="Training-Accuracy">

It can be read from the graph that **the categorical accuracy on validation set is around 0.96**

#### Testing

First is the classification report from sklearn:

```bash
              precision    recall  f1-score   support

     cyclist       0.97      0.97      0.97      1600
        misc       0.90      0.80      0.84      1157
  pedestrian       0.96      0.97      0.96      2225
     vehicle       0.95      0.98      0.96      3034

    accuracy                           0.95      8016
   macro avg       0.94      0.93      0.93      8016
weighted avg       0.95      0.95      0.95      8016
```

And the confusion matrix:

<img src="doc/training-confusion-matrix.png" alt="Training-Confusion Matrix">

The following conclusions can be drawn from above data:

* **The Trained Model Generalizes Very Well on Test Set** 

    Because the categorical accuracy on validation and test sets are comparable.

* **The Model's Performance Bottleneck is on the Misc Classes** 

    The model has poor accuracies for misc class compared with the three types of traffic participants. This is caused by the great intrinsic variety of misc class due to its generation nature.

