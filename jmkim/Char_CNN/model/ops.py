import tensorflow as tf
from tensorflow.python.keras import layers
import numpy as np

class ConvolutionLayer(tf.keras.layers.Layer):
    def __init__(self, vocab, filter):
        super(ConvolutionLayer, self).__init__()
        self._lookup = tf.Variable(name='lookup', initial_value=np.array(list(vocab.token_to_idx.values())), trainable=True)
        # Conv1D
        self._kernel_7_conv = layers.Conv1D(filters=filter, activation=tf.nn.relu, kernel_size=7)
        self._kernel_3_conv = layers.Conv1D(filters=filter, activation=tf.nn.relu, kernel_size=3)
        self._maxpooling = layers.MaxPool1D(pool_size=3, strides=1, padding='same')

    def call(self, x):
        print(x.shape)
        lookup = tf.nn.embedding_lookup(params=self._lookup, ids=x)
        print(lookup)
        #x = tf.reshape(x, [128, 1014, 1])
        #print(x.shape)
        conv = self._kernel_7_conv(lookup)
        conv = self._maxpooling(conv)
        print(conv.shape)
        conv = self._kernel_7_conv(conv)
        conv = self._maxpooling(conv)
        print(conv.shape)
        conv = self._kernel_3_conv(conv)
        conv = self._kernel_3_conv(conv)
        conv = self._kernel_3_conv(conv)
        conv = self._kernel_3_conv(conv)
        conv = self._maxpooling(conv)
        print(conv.shape)
        #return layers.Flatten(conv)
        return conv


class Classifier(tf.keras.layers.Layer):
    def __init__(self, dropout, dim, classes):
        super(Classifier, self).__init__()
        self._dropout = tf.keras.layers.Dropout(dropout)
        self._dense = tf.keras.layers.Dense(dim, activation='relu')
        self._outDense = tf.keras.layers.Dense(classes)

    def call(self, x):
        output = self._dense(x)
        output = self._dropout(output)
        output = self._dense(output)
        output = self._dropout(output)
        output = self._outDense(output)
        return output
