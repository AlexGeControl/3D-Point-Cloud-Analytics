#!/opt/conda/envs/05-deep-classification/bin/python

import os
import sys
import datetime

sys.path.insert(0, './')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from preprocess import ModelNet40Dataset

import numpy as np

import tensorflow as tf
from tensorflow import keras
from models.cls_msg_model import CLS_MSG_Model
from models.cls_ssg_model import CLS_SSG_Model

from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sn
import matplotlib.pyplot as plt

tf.random.set_seed(1234)

def load_dataset(input_filename, batch_size):
	""" 
	Load dataset

	Parameters
	----------
	input_filename: str 
		Filename of dataset as TFRecord.
	batch_size: int 
		Mini-batch size.

	"""
	assert os.path.isfile(input_filename), '[ModelNet40 Train] ERROR--dataset path not found'

	dataset = tf.data.TFRecordDataset(input_filename)
	dataset = dataset.shuffle(
		buffer_size=1024, 
		reshuffle_each_iteration=True
	)
	dataset = dataset.map(ModelNet40Dataset.deserialize)
	dataset = dataset.map(ModelNet40Dataset.preprocess)
	dataset = dataset.batch(batch_size, drop_remainder=True)

	return dataset


def train(config):
	""" 
	Build network

	Parameters
	----------
	config: dict 
		Model training configuration

	"""

	# load dataset:
	training_data = load_dataset(config['training_data'], config['batch_size'])
	validation_data = load_dataset(config['validation_data'], config['batch_size'])

	# init model:
	if config['msg'] == True:
		model = CLS_MSG_Model(config['batch_size'], config['num_classes'], config['batch_normalization'])
	else:
		model = CLS_SSG_Model(config['batch_size'], config['num_classes'], config['batch_normalization'])

	# enable early stopping:
	callbacks = [
		keras.callbacks.EarlyStopping(
			'val_sparse_categorical_accuracy', min_delta=0.01, patience=10),
		keras.callbacks.TensorBoard(
			'./logs/{}'.format(config['log_dir']), update_freq=50),
		keras.callbacks.ModelCheckpoint(
			'./logs/{}/model/weights.ckpt'.format(config['log_dir']), 'val_sparse_categorical_accuracy', save_best_only=True)
	]

	model.build(input_shape=(config['batch_size'], ModelNet40Dataset.N, ModelNet40Dataset.d+ModelNet40Dataset.C))
	print(model.summary())

	model.compile(
		optimizer=keras.optimizers.Adam(config['lr']),
		loss=keras.losses.SparseCategoricalCrossentropy(),
		metrics=[keras.metrics.SparseCategoricalAccuracy()]
	)

	model.fit(
		training_data,
		validation_data = validation_data,
		validation_steps = 20,
		validation_freq = 1,
		callbacks = callbacks,
		epochs = 100,
		verbose = 1
	)

def predict(config):
	""" 
	Load trained network and make predictions

	Parameters
	----------
	config: dict 
		Model training configuration

	"""
	# load dataset:
	data = load_dataset(config['test_data'], config['batch_size'])

	# init model:
	if config['msg'] == True:
		model = CLS_MSG_Model(config['batch_size'], config['num_classes'], config['batch_normalization'])
	else:
		model = CLS_SSG_Model(config['batch_size'], config['num_classes'], config['batch_normalization'])

	# load params:
	model.load_weights(config['checkpoint_path'])

	y_truths = []
	y_preds = []
	for X, y in data:
		y_truths.append(y.numpy().flatten())
		y_preds.append(np.argmax(model(X), axis=1))
	
	y_truth = np.hstack(y_truths)
	y_pred = np.hstack(y_preds)

	# get decoder
	decoder = ModelNet40Dataset(input_dir='/workspace/data/modelnet40_normal_resampled').get_decoder()

	plt.figure(figsize = (10,10))
	sn.heatmap(confusion_matrix(y_truth, y_pred), annot=True)
	plt.show()

	print(
		classification_report(
			y_truth, y_pred, 
			target_names=[
				decoder[i] for i in range(40)
			]
		)
	)

if __name__ == '__main__':
	# config = {
	# 	'training_data' : 'data/modelnet40_with_normal_train.tfrecord',
	# 	'validation_data' : 'data/modelnet40_with_normal_validate.tfrecord',
	# 	'test_data' : 'data/modelnet40_with_normal_test.tfrecord',
	# 	'log_dir' : 'msg_1',
	# 	'batch_size' : 16,
	# 	'lr' : 0.001,
	# 	'num_classes' : 40,
	# 	'msg' : True,
	# 	'batch_normalization' : False
	# }

	# train(config)

	config = {
		'test_data' : 'data/modelnet40_with_normal_test.tfrecord',
		'msg' : True,
		'batch_size' : 16,
		'num_classes' : 40,
		'batch_normalization' : False,
		'checkpoint_path' : 'logs/msg_1/model/weights.ckpt',
	}

	predict(config)