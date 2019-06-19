import tensorflow as tf
from model.ops import ConvolutionLayer, Classifier

from configs import FLAGS


class CharCNN(tf.keras.Model):
    def __init__(self, vocab, classes, dim):
        super(CharCNN, self).__init__()
        self._conv = ConvolutionLayer(vocab=vocab)
        self._classifi = Classifier(classes=classes, dropout=FLAGS.dropout)

    def call(self, x):
        conv = self._conv(x)
        classifi = self._classifi(conv)

        return classifi
