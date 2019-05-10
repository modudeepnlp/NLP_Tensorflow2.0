import tensorflow as tf
from model.ops import MultiChannelEmbedding, ConvolutionLayer

from configs import FLAGS


class SenCNN(tf.keras.Model):
    def __init__(self, vocab, max_len, dim):
        super(SenCNN, self).__init__()
        self._embedding = MultiChannelEmbedding(vocab, max_len, dim)
        self._conv = ConvolutionLayer(FLAGS.length, dim)

    def call(self, x):
        feature_map = self._embedding(x)
        feature_map = self._conv(feature_map)
        return feature_map
