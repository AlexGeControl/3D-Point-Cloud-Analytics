#!/opt/conda/envs/kitti-detection-pipeline/bin/python

# preprocess.py
#     Convert object classification dataset in TXT into tfrecords

import argparse
import glob
import os
import shutil

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf

import progressbar

class KITTIPCDClassificationDataset:
    """
    KITTI point cloud classification dataset for deep learning object classification

    Parameters
    ----------
    input_dir: str 
        Directory path of dataset.
    filename_labels: str
        Filename of shape labels. Defaults to 'object_names.txt'.
    filename_train: str
        Filename of training set specification
    filename_test
        Filename of testing set specification

    Attributes
    ----------
    
    """
    N = 64
    d = 3
    C = 3

    def __init__(
        self, 
        input_dir,
        N = 64,
        filename_labels = 'object_names.txt',
        filename_train = 'train.txt',
        size_validate = 0.20,
        filename_test = 'test.txt',
        random_seed=42
    ):
        # I/O spec:
        self.__input_dir = input_dir
        # load labels:
        self.__labels, self.__encoder, self.__decoder = self.__load_labels(filename_labels)
        # num. of observations in point cloud:
        self.__N = N
        # load training set:
        self.__train = np.asarray(
            self.__load_examples(filename_train)
        )
        # create validation set:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=size_validate, random_state=random_seed)
        for fit_index, validate_index in sss.split(            
            self.__train, 
            # labels:
            [t.split('_')[0] for t in self.__train]
        ):
            self.__fit, self.__validate = self.__train[fit_index], self.__train[validate_index]
        # load test set:
        self.__test = self.__load_examples(filename_test)

        # remove orders:
        random.seed(random_seed)
        random.shuffle(self.__fit)
        random.shuffle(self.__validate)
        random.shuffle(self.__test)

    def __load_labels(self, filename_labels):
        """ 
        Load labels

        Parameters
        ----------
        filename_labels: str 
            Filename of dataset labels.

        """
        (labels, encoder, decoder) = (None, None, None)

        with open(os.path.join(self.__input_dir, filename_labels)) as f:
            labels = [l.strip() for l in f.readlines()]

        encoder = {label: id for id, label in enumerate(labels)}
        decoder = {id: label for id, label in enumerate(labels)}

        return (labels, encoder, decoder)

    def __load_examples(self, filename_split):
        """ 
        Load examples

        Parameters
        ----------
        filename_split: str 
            Filename of split specification.

        """
        examples = None

        with open(os.path.join(self.__input_dir, filename_split)) as f:
            examples = [e.strip() for e in f.readlines()]

        return examples

    def __get_label(self, filename_example):
        """ 
        Get label of example

        Parameters
        ----------
        filename_example: str 
            Short filename of example.

        """
        label, idx = filename_example.rsplit('_', 1)

        return label, idx

    def __get_filename_point_cloud(self, filename_example):
        """ 
        Get relative path of example

        Parameters
        ----------
        filename_example: str 
            Short filename of example.

        """
        # get label:
        label, idx = self.__get_label(filename_example)
        
        # generate relative filename:
        filename_point_cloud = os.path.join(
            self.__input_dir, label, idx
        )
        filename_point_cloud = f'{filename_point_cloud}.txt'

        return filename_point_cloud

    def __write(self, filename_examples, filename_tfrecord):
        """ 
        Save split to TFRecord

        Parameters
        ----------
        filename_examples: str 
            Filenames of split examples.
        filename_tfrecord: str 
            Filename of output TFRecord.

        """
        with tf.io.TFRecordWriter(filename_tfrecord) as writer:
            for filename_example in progressbar.progressbar(filename_examples):
                # parse point cloud:
                filename_point_cloud = self.__get_filename_point_cloud(filename_example)
                df_point_cloud_with_normal = pd.read_csv(
                    filename_point_cloud, 
                    header=None, names=['x', 'y', 'z', 'nx', 'ny', 'nz']
                )
                # get label:
                label, _ = self.__get_label(filename_example)
                # format:
                xyz = df_point_cloud_with_normal[['x', 'y', 'z']].values.astype(np.float32)
                points = df_point_cloud_with_normal[['nx', 'ny', 'nz']].values.astype(np.float32)
                label_id = self.__encoder[label]
                # write to tfrecord:
                serialized_example = KITTIPCDClassificationDataset.serialize(xyz, points, label_id)
                writer.write(serialized_example)

    def get_labels(self):
        return self.__labels

    def get_encoder(self):
        return self.__encoder

    def get_decoder(self):
        return self.__decoder

    @staticmethod
    def __bytes_feature(value):
        """
        Returns a bytes_list from a string / byte.
        """
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def __float_feature(value):
        """
        Returns a float_list from a float / double.
        """
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def __floats_feature(value):
        """
        Returns a float_list from a numpy.ndarray.
        """
        return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))

    @staticmethod
    def __int64_feature(value):
        """
        Returns an int64_list from a bool / enum / int / uint.
        """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def serialize(xyz, points, label):
        """ 
        Serialize 

        Parameters
        ----------
        xyz: numpy.ndarray 
            Point cloud coordinates.
        points: numpy.ndarray 
            Point cloud features.
        label: int 
            Shape ID.

        """
        N_p, d = xyz.shape
        N_f, C = points.shape

        assert N_p == N_f, '[KITTI Point Cloud Classification Dataset (With Normal)] ERROR--Dimensions mismatch: xyz & points.'

        feature = {
            'xyz': KITTIPCDClassificationDataset.__floats_feature(xyz),
            'points': KITTIPCDClassificationDataset.__floats_feature(points),
            'label': KITTIPCDClassificationDataset.__int64_feature(label),
            'N': KITTIPCDClassificationDataset.__int64_feature(N_p),
            'd': KITTIPCDClassificationDataset.__int64_feature(d),
            'C': KITTIPCDClassificationDataset.__int64_feature(C)
        }

        example = tf.train.Example(
            features=tf.train.Features(feature=feature)
        )
        
        return example.SerializeToString()

    @staticmethod
    def deserialize(serialized_example):
        """ 
        Serialize 

        Parameters
        ----------
        serialized_example: str
            TFRecird serialized example

        """
        feature_description = {
            'xyz': tf.io.FixedLenFeature(
                [KITTIPCDClassificationDataset.N * KITTIPCDClassificationDataset.d], 
                tf.float32
            ),
            'points': tf.io.FixedLenFeature(
                [KITTIPCDClassificationDataset.N * KITTIPCDClassificationDataset.C], 
                tf.float32
            ),
            'label': tf.io.FixedLenFeature([1], tf.int64),
            'N': tf.io.FixedLenFeature([1], tf.int64),
            'd': tf.io.FixedLenFeature([1], tf.int64),
            'C': tf.io.FixedLenFeature([1], tf.int64),
        }

        example = tf.io.parse_single_example(
            serialized_example, 
            feature_description
        )
        
        return example 
        
    @staticmethod
    def preprocess(example):
        """ 
        Serialize 

        Parameters
        ----------
        serialized_example: str
            TFRecird serialized example

        """
        xyz = example['xyz']
        points = example['points']
        label = example['label']
        N = example['N']
        d = example['d']
        C = example['C']
        
        # format:
        xyz = tf.reshape(xyz, (KITTIPCDClassificationDataset.N, KITTIPCDClassificationDataset.d))
        points = tf.reshape(points, (KITTIPCDClassificationDataset.N, KITTIPCDClassificationDataset.C))

        # center to zero:
        xyz -= tf.reduce_mean(xyz, axis=0)

        # remove order in point cloud:
        indices = tf.range(start=0, limit=KITTIPCDClassificationDataset.N, dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)

        # use surface normals:
        features = tf.gather(
            tf.concat([xyz, points], 1), 
            shuffled_indices
        )

        return features, label

    def write(self, output_dir, output_name):
        """ 
        Serialize 

        Parameters
        ----------
        output_name: str
            Output TFRecord name

        """
        # init output root dir:
        output_dir = os.path.join(output_dir, 'tf_dataset')
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir) 
        os.makedirs(output_dir)

        print('[KITTI Point Cloud Classification Dataset (With Normal)]: Write training set...')        
        self.__write(
            self.__fit, 
            os.path.join(output_dir, f'{output_name}_train.tfrecord')
        )

        print('[KITTI Point Cloud Classification Dataset (With Normal)]: Write validation set...')  
        self.__write(
            self.__validate, 
            os.path.join(output_dir, f'{output_name}_validate.tfrecord')
        )

        print('[KITTI Point Cloud Classification Dataset (With Normal)]: Write testing set...')  
        self.__write(
            self.__test, 
            os.path.join(output_dir, f'{output_name}_test.tfrecord')
        )

def get_arguments():
    """ 
    Get command-line arguments

    """
    # init parser:
    parser = argparse.ArgumentParser("Format Open3D with normal in TXT into Open3D PCD.")

    # add required and optional groups:
    required = parser.add_argument_group('Required')

    # add required:
    required.add_argument(
        "-i", dest="input", help="Input path.",
        required=True, type=str
    )
    required.add_argument(
        "-o", dest="output_name", help="Output TFRecord name.",
        required=True, type=str
    )

    # parse arguments:
    return parser.parse_args()


if __name__ == "__main__":
    # parse command line arguments
    args = get_arguments()

    # read dataset:
    kitti_pcd_classification_dataset = KITTIPCDClassificationDataset(
        args.input
    )

    # convert into TFRecord
    kitti_pcd_classification_dataset.write(
        args.input, args.output_name
    )