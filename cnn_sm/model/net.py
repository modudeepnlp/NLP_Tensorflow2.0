import tensorflow as tf
from tensorflow.keras import layers
from cnn_sm.model.ops import MultiChannelEmbedding, ConvolutionLayer, MaxPooling
from gluonnlp import Vocab


class SenCNN(tf.keras.Model):
    def __init__(self, num_classes: int, vocab: Vocab) -> None:
        super(SenCNN, self).__init__()
        self._embedding = MultiChannelEmbedding(vocab)
        self._convolution = ConvolutionLayer(300)
        self._pooling = MaxPooling()
        self._dropout = layers.Dropout(0.5)
        self._fc = layers.Dense(units=num_classes, activation='softmax')

    def call(self, x: tf.Tensor) -> tf.Tensor:
        fmap = self._embedding(x)
        fmap = self._convolution(fmap)
        feature = self._pooling(fmap)
        feature = self._dropout(feature)
        score = self._fc(feature)
        return score