import tensorflow as tf
from tensorflow.python.keras import layers
import numpy as np

from configs import FLAGS


class ConvolutionLayer(tf.keras.layers.Layer):
    def __init__(self, vocab, trainable):
        super(ConvolutionLayer, self).__init__()
        self._lookup = layers.Embedding(len(vocab), FLAGS.embedding_dim, input_length=FLAGS.length, trainable=trainable)  # [128, 256, 1014]
        # Conv1D
        self._kernel_7_conv = layers.Conv1D(filters=FLAGS.embedding_dim, activation=tf.nn.relu, kernel_size=7)
        self._kernel_3_conv = layers.Conv1D(filters=FLAGS.embedding_dim, activation=tf.nn.relu, kernel_size=3)
        self._maxpooling = layers.GlobalMaxPool1D()

    def call(self, x):
        #print(x.shape)  # [128, 1014]
        lookup = self._lookup(x)
        #print(lookup.shape)  # [128,1014, 256]
        conv = self._kernel_7_conv(lookup)
        #print(conv.shape)  # [128, 1008, 256]
        conv = self._maxpooling(conv)
        #print(conv.shape)  # [128, 256]
        conv = self._lookup(conv)
        conv = self._kernel_7_conv(conv)
        conv = self._maxpooling(conv)
        #print(conv.shape)
        conv = self._lookup(conv)
        conv = self._kernel_3_conv(conv)
        conv = self._kernel_3_conv(conv)
        conv = self._kernel_3_conv(conv)
        conv = self._kernel_3_conv(conv)
        conv = self._maxpooling(conv)
        #print(conv.shape)
        # return layers.Flatten(conv)
        return conv


class Classifier(tf.keras.layers.Layer):
    def __init__(self, dropout, dim, classes):
        super(Classifier, self).__init__()
        self._dropout = tf.keras.layers.Dropout(dropout)
        self._dense = tf.keras.layers.Dense(FLAGS.embedding_dim, activation='relu')
        self._outDense = tf.keras.layers.Dense(classes)

    def call(self, x):
        output = self._dense(x)
        output = self._dropout(output)
        output = self._dense(output)
        output = self._dropout(output)
        output = self._outDense(output)
        return output
