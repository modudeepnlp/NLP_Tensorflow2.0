import tensorflow as tf
from tensorflow.python.keras import layers
import numpy as np

from configs import FLAGS


class ConvolutionLayer(tf.keras.layers.Layer):
	def __init__(self, vocab):
		super(ConvolutionLayer, self).__init__()
		self._lookup = layers.Embedding(len(vocab), FLAGS.embedding_dim, input_length=FLAGS.length)  # [128, 256, 1014]
		self._maxpooling = layers.MaxPool1D(pool_size=3, strides=3)
	def call(self, x):
		#print(x.shape)  # [32, 1014]
		lookup = self._lookup(x)
		#print(lookup.shape)  # [32, 1014, 256]
		conv = layers.Conv1D(filters=FLAGS.embedding_dim, activation=tf.nn.relu, kernel_size=7)(lookup)
		#print(conv.shape)  # [32, 1008, 256]
		conv = self._maxpooling(conv)
		#print(conv.shape)  # [32, 336, 256]
		conv = layers.Conv1D(filters=FLAGS.embedding_dim, activation=tf.nn.relu, kernel_size=7)(conv)
		conv = self._maxpooling(conv)
		#print(conv.shape) # [32, 110, 256]
		conv = layers.Conv1D(filters=FLAGS.embedding_dim, activation=tf.nn.relu, kernel_size=3)(conv)
		conv = layers.Conv1D(filters=FLAGS.embedding_dim, activation=tf.nn.relu, kernel_size=3)(conv)
		conv = layers.Conv1D(filters=FLAGS.embedding_dim, activation=tf.nn.relu, kernel_size=3)(conv)
		conv = layers.Conv1D(filters=FLAGS.embedding_dim, activation=tf.nn.relu, kernel_size=3)(conv)
		conv = self._maxpooling(conv)
		#print(conv.shape) # [32, 34, 256]
		return layers.Flatten()(conv)


class Classifier(tf.keras.layers.Layer):
	def __init__(self, dropout, classes):
		super(Classifier, self).__init__()
		self._dropout = tf.keras.layers.Dropout(dropout)
		self._dropout1 = tf.keras.layers.Dropout(dropout)
		self._dense1 = tf.keras.layers.Dense(FLAGS.embedding_dim, activation=tf.nn.relu)
		self._dense2 = tf.keras.layers.Dense(FLAGS.embedding_dim, activation=tf.nn.relu)
		self._outDense = tf.keras.layers.Dense(classes, activation='softmax')

	def call(self, x):
		#print(x.shape) # [32, 8704]
		output = self._dense1(x)
		output = self._dropout(output)
		output = self._dense2(output)
		output = self._dropout1(output)
		output = self._outDense(output)
		return output
