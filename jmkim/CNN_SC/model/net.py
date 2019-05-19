import tensorflow as tf
from model.ops import MultiChannelEmbedding, ConvolutionLayer, MaxPooling

from configs import FLAGS


class SenCNN(tf.keras.Model):
    def __init__(self, vocab, max_len, dim, classes):
        super(SenCNN, self).__init__()
        self._embedding = MultiChannelEmbedding(vocab, max_len, dim)
        self._conv = ConvolutionLayer(FLAGS.length, dim)
        # maxpooling
        self._maxpooling = MaxPooling()

        self._dropout = tf.keras.layers.Dropout(.5)
        self._fc = tf.keras.layers.Dense(units=classes)

    def call(self, x):
        feature_map = self._embedding(x)
        feature_map = self._conv(feature_map)
        feature = self._maxpooling(feature_map)
        feature = self._dropout(feature)
        score = self._fc(feature)

        return score
