import os
import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import MaxPool1D, Layer, BatchNormalization

from .cpp_modules import (
	farthest_point_sample,
	gather_point,
	query_ball_point,
	group_point,
	knn_point,
	three_nn,
	three_interpolate
)


def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):

	new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
	if knn:
		_,idx = knn_point(nsample, xyz, new_xyz)
	else:
		idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
	grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
	grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
	if points is not None:
		grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
		if use_xyz:
			new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
		else:
			new_points = grouped_points
	else:
		new_points = grouped_xyz

	return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):

	batch_size = xyz.get_shape()[0]
	nsample = xyz.get_shape()[1]

	new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)

	idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
	grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
	if points is not None:
		if use_xyz:
			new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
		else:
			new_points = points
		new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)
	else:
		new_points = grouped_xyz
	return new_xyz, new_points, idx, grouped_xyz


class Conv2d(Layer):

	def __init__(self, filters, strides=[1, 1], activation=tf.nn.relu, padding='VALID', initializer='glorot_normal', bn=False):
		super(Conv2d, self).__init__()

		self.filters = filters
		self.strides = strides
		self.activation = activation
		self.padding = padding
		self.initializer = initializer
		self.bn = bn

	def build(self, input_shape):

		self.w = self.add_weight(
			shape=(1, 1, input_shape[-1], self.filters),
			initializer=self.initializer,
			trainable=True,
			name='pnet_conv'
		)

		if self.bn: self.bn_layer = BatchNormalization()

		super(Conv2d, self).build(input_shape)

	def call(self, inputs, training=True):

		points = tf.nn.conv2d(inputs, filters=self.w, strides=self.strides, padding=self.padding)

		if self.bn: points = self.bn_layer(points, training=training)

		if self.activation: points = self.activation(points)

		return points
