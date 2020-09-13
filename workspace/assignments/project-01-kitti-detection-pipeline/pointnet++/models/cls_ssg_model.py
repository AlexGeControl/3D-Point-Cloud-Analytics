import os
import sys

sys.path.insert(0, './')

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

from pnet2_layers.layers import Pointnet_SA, Pointnet_SA_MSG


class CLS_SSG_Model(Model):

	def __init__(self, batch_size, num_classes, bn=False, activation=tf.nn.relu):
		super(CLS_SSG_Model, self).__init__()

		self.activation = activation
		self.batch_size = batch_size
		self.num_classes = num_classes
		self.bn = bn
		self.keep_prob = 0.5

		self.kernel_initializer = 'glorot_normal'
		self.kernel_regularizer = None

		self.init_network()


	def init_network(self):

		self.layer1 = Pointnet_SA(
			npoint=512, radius=0.2,
			nsample=32,
			mlp=[64, 64, 128],
			group_all=False,
			activation=self.activation,
			bn = self.bn
		)

		self.layer2 = Pointnet_SA(
			npoint=128,
			radius=0.4,
			nsample=64,
			mlp=[128, 128, 256],
			group_all=False,
			activation=self.activation,
			bn = self.bn
		)

		self.layer3 = Pointnet_SA(
			npoint=None,
			radius=None,
			nsample=None,
			mlp=[256, 512, 1024],
			group_all=True,
			activation=self.activation,
			bn = self.bn
		)

		self.dense1 = Dense(512, activation=self.activation)
		self.dropout1 = Dropout(self.keep_prob)

		self.dense2 = Dense(128, activation=self.activation)
		self.dropout2 = Dropout(self.keep_prob)

		self.dense3 = Dense(self.num_classes, activation=tf.nn.softmax)


	def forward_pass(self, input, training):

		xyz, points = self.layer1(input, None, training=training)
		xyz, points = self.layer2(xyz, points, training=training)
		xyz, points = self.layer3(xyz, points, training=training)

		net = tf.reshape(points, (self.batch_size, -1))

		net = self.dense1(net)
		net = self.dropout1(net)

		net = self.dense2(net)
		net = self.dropout2(net)

		pred = self.dense3(net)

		return pred


	def train_step(self, input):

		with tf.GradientTape() as tape:

			pred = self.forward_pass(input[0], True)
			loss = self.compiled_loss(input[1], pred)
		
		gradients = tape.gradient(loss, self.trainable_variables)
		self.optimizer.apply_gradients(
			zip(gradients, self.trainable_variables))

		self.compiled_metrics.update_state(input[1], pred)

		return {m.name: m.result() for m in self.metrics}


	def test_step(self, input):

		pred = self.forward_pass(input[0], False)
		loss = self.compiled_loss(input[1], pred)

		self.compiled_metrics.update_state(input[1], pred)

		return {m.name: m.result() for m in self.metrics}


	def call(self, input, training=False):

		return self.forward_pass(input, training)
