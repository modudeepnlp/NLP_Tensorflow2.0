import tensorflow as tf
from tensorflow.python.keras import layers


class ConvolutionLayer(tf.keras.layers.Layer):
    def __init__(self, filter):
        super(ConvolutionLayer, self).__init__()

        # Conv1D
        self._kernel_7_conv = layers.Conv1D(filters=filter, activation=tf.nn.relu, kernel_size=7)
        self._kernel_7_same_conv = layers.Conv1D(filters=filter, activation=tf.nn.relu, kernel_size=7, padding='same')
        self._kernel_3_conv = layers.Conv1D(filters=filter, activation=tf.nn.relu, kernel_size=3)
        self._kernel_3_same_conv = layers.Conv1D(filters=filter, activation=tf.nn.relu, kernel_size=3, padding='same')
        self._maxpooling = layers.MaxPool1D(3)

    def call(self, x):
        conv = self._kernel_7_conv(x)
        conv = self._maxpooling(conv)
        conv = self._kernel_7_same_conv(conv)
        conv = self._maxpooling(conv)
        conv = self._kernel_3_same_conv(conv)
        conv = self._kernel_3_same_conv(conv)
        conv = self._kernel_3_same_conv(conv)
        conv = self._kernel_3_same_conv(conv)
        conv = self._maxpooling(conv)

        return layers.Flatten(conv)


class Classifier(tf.keras.layers.Layer):
    def __init__(self, dropout, dim, classes):
        super(Classifier, self).__init__()
        self._dropout = tf.keras.layers.Dropout(dropout)
        self._dense = tf.keras.layers.Dense(dim, activation='relu')
        self._outDense = tf.keras.layers.Dense(classes)

    def call(self):
        return
