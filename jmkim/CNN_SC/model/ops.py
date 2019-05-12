import tensorflow as tf
from tensorflow.keras import layers

from gluonnlp import Vocab
from configs import FLAGS


class MultiChannelEmbedding(tf.keras.Model):
    def __init__(self, vocab, max_len, dim):
        super(MultiChannelEmbedding, self).__init__()

        self._vocab_len = len(vocab)
        self._max_len = max_len
        self._dim = dim
        self._embedding = vocab.embedding.idx_to_vec.asnumpy()
        self._static = tf.Variable(name='static', initial_value=self._embedding, trainable=True)
        self._non_static = tf.Variable(name='non_static', initial_value=self._embedding, trainable=False)

    def call(self, x):
        static_batch = tf.nn.embedding_lookup(params=self._static, ids=x)
        non_static_batch = tf.nn.embedding_lookup(params=self._non_static, ids=x)
        return static_batch, non_static_batch


class ConvolutionLayer(tf.keras.Model):
    def __init__(self, max_len, dim):
        super(ConvolutionLayer, self).__init__()

        self._max_len = max_len
        self._dim = dim

        self._tri_gram = layers.Conv1D(filters= 100 // 3, activation=tf.nn.relu, kernel_size=3)
        self._tetra_gram = layers.Conv1D(filters=100 // 4, activation=tf.nn.relu, kernel_size=4)
        self._penta_gram = layers.Conv1D(filters=100 // 5, activation=tf.nn.relu, kernel_size=5)


    def call(self, x):
        static, non_static = x

        cnn_3 = self._tri_gram(static)
        cnn_4 = self._tetra_gram(static)
        cnn_5 = self._penta_gram(static)

        non_cnn_3 = self._tri_gram(non_static)
        non_cnn_4 = self._tetra_gram(non_static)
        non_cnn_5 = self._penta_gram(non_static)

        tri = cnn_3 + non_cnn_3
        tetra = cnn_4 + non_cnn_4
        penta = cnn_5 + non_cnn_5
        return tri, tetra, penta
