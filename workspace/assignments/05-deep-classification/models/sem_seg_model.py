import os
import sys

sys.path.insert(0, './')

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout

from pnet2_layers.layers import Pointnet_SA, Pointnet_FP


class SEM_SEG_Model(Model):

	def __init__(self, batch_size, num_classes, bn=False, activation=tf.nn.relu):
		super(SEM_SEG_Model, self).__init__()

		self.activation = activation
		self.batch_size = batch_size
		self.keep_prob = 0.5
		self.num_classes = num_classes
		self.bn = bn

		self.kernel_initializer = 'glorot_normal'
		self.kernel_regularizer = None

		self.init_network()


	def init_network(self):

		self.sa_1 = Pointnet_SA(
			npoint=1024,
			radius=0.1,
			nsample=32,
			mlp=[32, 32, 64],
			group_all=False,
			activation=self.activation,
			bn = self.bn
		)

		self.sa_2 = Pointnet_SA(
			npoint=256,
			radius=0.2,
			nsample=32,
			mlp=[64, 64, 128],
			group_all=False,
			activation=self.activation,
			bn = self.bn
		)

		self.sa_3 = Pointnet_SA(
			npoint=64,
			radius=0.4,
			nsample=32,
			mlp=[128, 128, 256],
			group_all=False,
			activation=self.activation,
			bn = self.bn
		)

		self.sa_4 = Pointnet_SA(
			npoint=16,
			radius=0.8,
			nsample=32,
			mlp=[256, 256, 512],
			group_all=False,
			activation=self.activation,
			bn = self.bn
		)

		self.fp_1 = Pointnet_FP(
			mlp = [256, 256],
			activation = self.activation,
			bn = self.bn
		)

		self.fp_2 = Pointnet_FP(
			mlp = [256, 256],
			activation = self.activation,
			bn = self.bn
		)

		self.fp_3 = Pointnet_FP(
			mlp = [256, 128],
			activation = self.activation,
			bn = self.bn
		)

		self.fp_4 = Pointnet_FP(
			mlp = [128, 128, 128],
			activation = self.activation,
			bn = self.bn
		)


		self.dense1 = Dense(128, activation=self.activation)

		self.dropout1 = Dropout(self.keep_prob)

		self.dense2 = Dense(self.num_classes, activation=tf.nn.softmax)


	def forward_pass(self, input, training):

		l0_xyz = input
		l0_points = None

		l1_xyz, l1_points = self.sa_1(l0_xyz, l0_points, training=training)
		l2_xyz, l2_points = self.sa_2(l1_xyz, l1_points, training=training)
		l3_xyz, l3_points = self.sa_3(l2_xyz, l2_points, training=training)
		l4_xyz, l4_points = self.sa_4(l3_xyz, l3_points, training=training)

		l3_points = self.fp_1(l3_xyz, l4_xyz, l3_points, l4_points, training=training)
		l2_points = self.fp_2(l2_xyz, l3_xyz, l2_points, l3_points, training=training)
		l1_points = self.fp_3(l1_xyz, l2_xyz, l1_points, l2_points, training=training)
		l0_points = self.fp_4(l0_xyz, l1_xyz, l0_points, l1_points, training=training)

		net = self.dense1(l0_points)
		net = self.dropout1(net)
		pred = self.dense2(net)

		return pred


	def train_step(self, input):

		with tf.GradientTape() as tape:

			pred = self.forward_pass(input[0], True)
			loss = self.compiled_loss(input[1], pred)
		
		gradients = tape.gradient(loss, self.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

		self.compiled_metrics.update_state(input[1], pred)

		return {m.name: m.result() for m in self.metrics}


	def test_step(self, input):

		pred = self.forward_pass(input[0], False)
		loss = self.compiled_loss(input[1], pred)

		self.compiled_metrics.update_state(input[1], pred)

		return {m.name: m.result() for m in self.metrics}


	def call(self, input, training=False):

		return self.forward_pass(input, training)

