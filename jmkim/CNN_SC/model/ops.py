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

        self._tri_gram = layers.Conv2D(100, (3, self._dim), activation='relu', kernel_initializer='he_uniform',
                                       padding='valid')
        self._tri_max_pooling = layers.MaxPool2D(self._max_len - 3 + 1, 1)
        self._tetra_gram = layers.Conv2D(100, (4, self._dim), activation='relu', kernel_initializer='he_uniform',
                                         padding='valid')
        self._tetra_max_pooling = layers.MaxPool2D(self._max_len - 4 + 1, 1)
        self._penta_gram = layers.Conv2D(100, (5, self._dim), activation='relu', kernel_initializer='he_uniform',
                                         padding='valid')
        self._penta_max_pooling = layers.MaxPool2D(self._max_len - 5 + 1, 1)

        self._flatten = layers.Flatten()
        self._drop_out = layers.Dropout(0.5)
        self._dense_out = layers.Dense(FLAGS.classes, activation='sigmoid')

    def call(self, x):
        static, non_static = x
        # 여긴 다시 만들어야 하는 부분
        cnn_3 = self._tri_gram(static)
        max_3 = self._tri_max_pooling(cnn_3)
        cnn_4 = self._tri_gram(static)
        max_4 = self._tri_max_pooling(cnn_4)
        cnn_5 = self._tri_gram(static)
        max_5 = self._tri_max_pooling(cnn_5)

        concat = layers.concatenate([max_3, max_4, max_5])
        flatten = self._flatten(concat)
        drop_out = self._dropout(flatten)
        dense_out = self._dense_out(drop_out)

        return dense_out
