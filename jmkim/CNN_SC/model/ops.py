import tensorflow as tf
from tensorflow.keras import layers


class MultiChannelEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab):
        super(MultiChannelEmbedding, self).__init__()
        self._static = tf.Variable(name='static', initial_value=vocab.embedding.idx_to_vec.asnumpy(), trainable=False)
        self._non_static = tf.Variable(name='non_static', initial_value=vocab.embedding.idx_to_vec.asnumpy(),
                                       trainable=True)

    def call(self, x):
        static_batch = tf.nn.embedding_lookup(params=self._static, ids=x)
        non_static_batch = tf.nn.embedding_lookup(params=self._non_static, ids=x)
        return static_batch, non_static_batch


class ConvolutionLayer(tf.keras.layers.Layer):
    def __init__(self, filter):
        super(ConvolutionLayer, self).__init__()

        # Conv1D, Conv2D
        self._tri_gram = layers.Conv1D(filters=filter // 3, activation=tf.nn.relu, kernel_size=3)
        self._tetra_gram = layers.Conv1D(filters=filter // 3, activation=tf.nn.relu, kernel_size=4)
        self._penta_gram = layers.Conv1D(filters=filter // 3, activation=tf.nn.relu, kernel_size=5)

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


class MaxPooling(tf.keras.layers.Layer):
    def call(self, x):
        tri, tetra, penta = x
        # 여기 좀 해석할 것.
        concat = tf.concat([tf.reduce_max(tri, 1), tf.reduce_max(tetra, 1), tf.reduce_max(penta, 1)], axis=-1)
        return concat
