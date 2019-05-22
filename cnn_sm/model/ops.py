import tensorflow as tf
from tensorflow.keras import layers
import gluonnlp as nlp
from gluonnlp import Vocab
from typing import Tuple


class MultiChannelEmbedding(tf.keras.Model):
    def __init__(self, vocab=Vocab):
        super(MultiChannelEmbedding, self).__init__()
        self._embedding = vocab.embedding.idx_to_vec.asnumpy()
        self._non_static_embedding = tf.Variable(initial_value=self._embedding, trainable=True)
        self._static_embedding = tf.Variable(initial_value=self._embedding, trainable=False)

    def call(self, idx: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        non_static_embedding = tf.nn.embedding_lookup(self._non_static_embedding, idx)
        static_embedding = tf.nn.embedding_lookup(self._static_embedding, idx)
        return non_static_embedding, static_embedding


class ConvolutionLayer(tf.keras.Model):
    def __init__(self, filters: int = 300) -> None:
        super(ConvolutionLayer, self).__init__()
        self._tri_gram_ops = tf.keras.layers.Conv1D(filters=filters // 3, kernel_size=3, activation='relu')
        self._tetra_gram_ops = tf.keras.layers.Conv1D(filters=filters // 3, kernel_size=4,  activation='relu')
        self._penta_gram_ops = tf.keras.layers.Conv1D(filters=filters // 3, kernel_size=5, activation='relu')

    def call(self, x: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        non_static_embedding, static_embedding = x
        tri_fmap = self._tri_gram_ops(non_static_embedding) + self._tri_gram_ops(static_embedding)
        tetra_fmap = self._tetra_gram_ops(non_static_embedding) + self._tetra_gram_ops(static_embedding)
        penta_fmap = self._penta_gram_ops(non_static_embedding) + self._penta_gram_ops(static_embedding)
        return tri_fmap, tetra_fmap, penta_fmap


class MaxPooling(tf.keras.Model):
    def call(self, x: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
        tri_fmap, tetra_fmap, penta_fmap = x
        fmap = tf.concat([tf.reduce_max(tri_fmap, 1), tf.reduce_max(tetra_fmap, 1), tf.reduce_max(penta_fmap, 1)],
                         axis=-1)
        return fmap