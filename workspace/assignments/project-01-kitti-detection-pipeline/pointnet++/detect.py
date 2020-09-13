#!/opt/conda/envs/kitti-detection-pipeline/bin/python

# detect.py
#     1. read Velodyne point cloud measurements
#     2. perform ground and object segmentation on point cloud
#     3. predict object category using classification network

import os
import glob
import argparse
import sys
import progressbar
import datetime

sys.path.insert(0, './')
# disable TF log display:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# utils:
import utils.velodyne as measurement
import utils.visualization as visualization
import utils.kitti as output
# segmentation:
import segmentation
# prediction:
from preprocess import KITTIPCDClassificationDataset
import numpy as np
import scipy.spatial

import tensorflow as tf
from tensorflow import keras
from models.cls_msg_model import CLS_MSG_Model
from models.cls_ssg_model import CLS_SSG_Model

tf.random.set_seed(1234)

# open3d:
import open3d as o3d

def load_model(config):
    """
    Load pre-trained object classification network
    
    """
    # init model:
    if config['msg'] == True:
        model = CLS_MSG_Model(config['batch_size'], config['num_classes'], config['batch_normalization'])
    else:
        model = CLS_SSG_Model(config['batch_size'], config['num_classes'], config['batch_normalization'])

    # load params:
    model.load_weights(config['checkpoint_path'])

    return model

def preprocess(
    segmented_objects, object_ids,
    config
):
    """
    Preprocess for classification network
    """
    # parse config:
    points = np.asarray(segmented_objects.points)
    normals = np.asarray(segmented_objects.normals)
    num_objects = max(object_ids) + 1

    # result:
    X = []
    y = []
    for object_id in range(num_objects):
        # 1. only keep object with enough number of observations:
        if ((object_ids == object_id).sum() <= 4):
            continue
        
        # 2. only keep object within max radius distance:
        object_center = np.mean(points[object_ids == object_id], axis=0)[:2]
        if (np.sqrt((object_center*object_center).sum()) > config['max_radius_distance']):
            continue
        
        # 3. resample:
        points_ = np.copy(points[object_ids == object_id])
        normals_ = np.copy(normals[object_ids == object_id])
        N, _ = points_.shape

        weights = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(points_, 'euclidean')
        ).mean(axis = 0)
        weights /= weights.sum()
        
        idx = np.random.choice(
            np.arange(N), 
            size = (config['num_sample_points'], ), replace=True if config['num_sample_points'] > N else False,
            p = weights
        )

        # 4. translate to zero-mean:
        points_processed, normals_processed = points_[idx], normals_[idx]
        points_processed -= points_.mean(axis = 0)

        # format as numpy.ndarray:
        X.append(
            np.hstack(
                (points_processed, normals_processed)
            )
        )
        y.append(object_id)

    # format as tf dataset:
    X = np.asarray(X)
    y = np.asarray(y)

    # pad to batch size:
    N = len(y)
    if (N % config['batch_size'] != 0):
        num_repeat = config['batch_size'] - N % config['batch_size']

        X = np.vstack(
            (
                X, 
                np.repeat(
                    X[0], num_repeat, axis=0
                ).reshape(
                    (-1, config['num_sample_points'], 6)
                )
            )
        )
        y = np.hstack(
            (y, np.repeat(y[0], num_repeat))
        )

    # format as tensorflow dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.convert_to_tensor(X, dtype=tf.float32), 
            tf.convert_to_tensor(y, dtype=tf.int64)
        )
    )
    dataset = dataset.batch(batch_size=config['batch_size'], drop_remainder=True)

    return dataset, N


def predict(segmented_objects, object_ids, model, config):
    """ 
    Load classification network and predict surrounding object category

    Parameters
    ----------
    config: dict 
        Model training configuration

    """
    # prepare data:
    dataset, N = preprocess(segmented_objects, object_ids, config)

    # make predictions:
    predictions = {
        class_id: {} for class_id in range(config['num_classes'])
    }
    num_predicted = 0

    for X, y in dataset:
        # predict:
        prob_preds = model(X)
        ids = y.numpy()

        # add to prediction:
        for (object_id, class_id, confidence) in zip(
            # object ID:
            ids,
            # category:
            np.argmax(prob_preds, axis=1),
            # confidence:
            np.max(prob_preds, axis=1)
        ):
            predictions[class_id][object_id] = confidence
            num_predicted += 1
            
            # skip padded instances:
            if (num_predicted == N):
                break

    return predictions

def detect(
    dataset_dir, index,
    max_radius_distance, num_sample_points,
    debug_mode
):
    # 0. generate I/O paths:
    input_velodyne = os.path.join(dataset_dir, 'velodyne', f'{index:06d}.bin')
    input_params = os.path.join(dataset_dir, 'calib', f'{index:06d}.txt')
    output_label = os.path.join(dataset_dir, 'shenlan_pipeline_pred_2', 'data', f'{index:06d}.txt')

    # 1. read Velodyne measurements and calib params:
    point_cloud = measurement.read_measurements(input_velodyne)
    param = measurement.read_calib(input_params)

    # 2. segment ground and surrounding objects -- here discard intensity channel:
    segmented_ground, segmented_objects, object_ids = segmentation.segment_ground_and_objects(point_cloud[:, 0:3])

    # 3. predict object category using classification network:
    config = {
        # preprocess:
        'max_radius_distance': max_radius_distance,
        'num_sample_points': num_sample_points,
        # predict:
        'msg' : True,
        'batch_size' : 16,
        'num_classes' : 4,
        'batch_normalization' : False,
        'checkpoint_path' : 'logs/msg_1/model/weights.ckpt',
    }
    model = load_model(config)
    predictions = predict(segmented_objects, object_ids, model, config)
    
    # TODO: refactor decoder implementation
    decoder = KITTIPCDClassificationDataset(input_dir='/workspace/data/kitti_3d_object_classification_normal_resampled').get_decoder()

    # debug mode:
    if (debug_mode):
        # print detection results:
        for class_id in predictions:
            # show category:
            print(f'[{decoder[class_id]}]')
            # show instances:
            for object_id in predictions[class_id]:
                print(f'\t[Object ID]: {object_id}, confidence {predictions[class_id][object_id]:.2f}')

        # visualize:
        bounding_boxes = visualization.get_bounding_boxes(
            segmented_objects, object_ids, 
            predictions, decoder
        )
        o3d.visualization.draw_geometries(
            [segmented_ground, segmented_objects] + bounding_boxes
        )
    
    # 4. format output for KITTI offline evaluation tool:
    label = output.to_kitti_eval_format(
        segmented_objects, object_ids, param,
        predictions, decoder
    )
    label.to_csv(output_label, sep=' ', header=False, index=False)

def get_arguments():
    """ 
    Get command-line arguments

    """
    # init parser:
    parser = argparse.ArgumentParser("Perform two-stage object detection on KITTI dataset.")

    # add required and optional groups:
    required = parser.add_argument_group('Required')
    optional = parser.add_argument_group('Optional')

    # add required:
    required.add_argument(
        "-i", dest="input", help="Input path.",
        required=True, type=str
    )

    optional.add_argument(
        "-d", dest="debug_mode", help="When enabled, visualize the result. Defaults to False. \n",
        required=False, type=bool, default=False
    )
    optional.add_argument(
        "-r", dest="max_radius_distance", help="Maximum radius distance between object and Velodyne lidar. \nUsed for ROI definition. Defaults to 25.0. \nONLY used in 'generate' mode.",
        required=False, type=float, default=25.0
    )
    optional.add_argument(
        "-n", dest="num_sample_points", help="Number of sample points to keep for each object. \nDefaults to 64. \nONLY used in 'generate' mode.",
        required=False, type=int, default=64
    )

    # parse arguments:
    return parser.parse_args()


if __name__ == "__main__":
    # parse command line arguments
    args = get_arguments()

    for label in progressbar.progressbar(
        glob.glob(
            os.path.join(args.input, 'shenlan_pipeline_label_2', '*.txt')
        )
    ):
        # get index:
        index = int(
            os.path.splitext(
                os.path.basename(label)
            )[0]
        )

        # perform object detection:
        detect(
            args.input, index,
            args.max_radius_distance, args.num_sample_points,
            args.debug_mode
        )
