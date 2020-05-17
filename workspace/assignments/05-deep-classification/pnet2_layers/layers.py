import tensorflow as tf
from tensorflow.keras.layers import Layer, BatchNormalization

from . import utils


class Pointnet_SA(Layer):

	def __init__(
		self, npoint, radius, nsample, mlp, group_all=False, knn=False, use_xyz=True, activation=tf.nn.relu, bn=False
	):

		super(Pointnet_SA, self).__init__()

		self.npoint = npoint
		self.radius = radius
		self.nsample = nsample
		self.mlp = mlp
		self.group_all = group_all
		self.knn = False
		self.use_xyz = use_xyz
		self.activation = activation
		self.bn = bn

		self.mlp_list = []

	def build(self, input_shape):

		for i, n_filters in enumerate(self.mlp):
			self.mlp_list.append(utils.Conv2d(n_filters, activation=self.activation, bn=self.bn))

		super(Pointnet_SA, self).build(input_shape)

	def call(self, xyz, points, training=True):

		if points is not None:
			if len(points.shape) < 3:
				points = tf.expand_dims(points, axis=0)

		if self.group_all:
			nsample = xyz.get_shape()[1]
			new_xyz, new_points, idx, grouped_xyz = utils.sample_and_group_all(xyz, points, self.use_xyz)
		else:
			new_xyz, new_points, idx, grouped_xyz = utils.sample_and_group(
				self.npoint,
				self.radius,
				self.nsample,
				xyz,
				points,
				self.knn,
				use_xyz=self.use_xyz
			)

		for i, mlp_layer in enumerate(self.mlp_list):
			new_points = mlp_layer(new_points, training=training)

		new_points = tf.math.reduce_max(new_points, axis=2, keepdims=True)

		return new_xyz, tf.squeeze(new_points)


class Pointnet_SA_MSG(Layer):

	def __init__(
		self, npoint, radius_list, nsample_list, mlp, use_xyz=True, activation=tf.nn.relu, bn = False
	):

		super(Pointnet_SA_MSG, self).__init__()

		self.npoint = npoint
		self.radius_list = radius_list
		self.nsample_list = nsample_list
		self.mlp = mlp
		self.use_xyz = use_xyz
		self.activation = activation
		self.bn = bn

		self.mlp_list = []

	def build(self, input_shape):

		for i in range(len(self.radius_list)):
			tmp_list = []
			for i, n_filters in enumerate(self.mlp[i]):
				tmp_list.append(utils.Conv2d(n_filters, activation=self.activation, bn=self.bn))
			self.mlp_list.append(tmp_list)

		super(Pointnet_SA_MSG, self).build(input_shape)

	def call(self, xyz, points, training=True):

		if points is not None:
			if len(points.shape) < 3:
				points = tf.expand_dims(points, axis=0)

		new_xyz = utils.gather_point(xyz, utils.farthest_point_sample(self.npoint, xyz))

		new_points_list = []

		for i in range(len(self.radius_list)):
			radius = self.radius_list[i]
			nsample = self.nsample_list[i]
			idx, pts_cnt = utils.query_ball_point(radius, nsample, xyz, new_xyz)
			grouped_xyz = utils.group_point(xyz, idx)
			grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1])

			if points is not None:
				grouped_points = utils.group_point(points, idx)
				if self.use_xyz:
					grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
			else:
				grouped_points = grouped_xyz

			for i, mlp_layer in enumerate(self.mlp_list[i]):
				grouped_points = mlp_layer(grouped_points, training=training)

			new_points = tf.math.reduce_max(grouped_points, axis=2)
			new_points_list.append(new_points)

		new_points_concat = tf.concat(new_points_list, axis=-1)

		return new_xyz, new_points_concat


class Pointnet_FP(Layer):

	def __init__(
		self, mlp, activation=tf.nn.relu, bn=False
	):

		super(Pointnet_FP, self).__init__()

		self.mlp = mlp
		self.activation = activation
		self.bn = bn

		self.mlp_list = []


	def build(self, input_shape):

		for i, n_filters in enumerate(self.mlp):
			self.mlp_list.append(utils.Conv2d(n_filters, activation=self.activation, bn=self.bn))
		super(Pointnet_FP, self).build(input_shape)

	def call(self, xyz1, xyz2, points1, points2, training=True):

		if points1 is not None:
			if len(points1.shape) < 3:
				points1 = tf.expand_dims(points1, axis=0)
		if points2 is not None:
			if len(points2.shape) < 3:
				points2 = tf.expand_dims(points2, axis=0)

		dist, idx = utils.three_nn(xyz1, xyz2)
		dist = tf.maximum(dist, 1e-10)
		norm = tf.reduce_sum((1.0/dist),axis=2, keepdims=True)
		norm = tf.tile(norm,[1,1,3])
		weight = (1.0/dist) / norm
		interpolated_points = utils.three_interpolate(points2, idx, weight)

		if points1 is not None:
			new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
		else:
			new_points1 = interpolated_points
		new_points1 = tf.expand_dims(new_points1, 2)

		for i, mlp_layer in enumerate(self.mlp_list):
			new_points1 = mlp_layer(new_points1, training=training)

		new_points1 = tf.squeeze(new_points1)
		if len(new_points1.shape) < 3:
			new_points1 = tf.expand_dims(new_points1, axis=0)

		return new_points1
