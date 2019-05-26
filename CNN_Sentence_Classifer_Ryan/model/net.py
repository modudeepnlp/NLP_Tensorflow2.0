import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

class CNN(tf.keras.Model):

	def __init__(self, max_len, emb_dim, vocab_size):

		super(CNN, self).__init__()

		self.MAX_LEN = max_len
		self.EMB_DIM = emb_dim
		self.VOC_SIZE = vocab_size

		self._embedding = layers.Embedding(self.VOC_SIZE, self.EMB_DIM, input_length=self.MAX_LEN)
		self._reshape = layers.Reshape((self.MAX_LEN, self.EMB_DIM, 1))

		self._cnn_filter_3 = layers.Conv2D(100, kernel_size=(3, self.EMB_DIM), padding='valid',
		                                   kernel_initializer='normal', activation='relu') # filters, kernel size
		self._max_pool_3 = layers.MaxPooling2D((self.MAX_LEN - 3 + 1, 1), strides=(1,1), padding='valid')

		self._cnn_filter_4 = layers.Conv2D(100, (4, self.EMB_DIM), padding='valid',
		                                   kernel_initializer='normal', activation='relu') # filters, kernel size
		self._max_pool_4 = layers.MaxPooling2D((self.MAX_LEN - 4 + 1, 1), strides=(1,1), padding='valid')

		self._cnn_filter_5 = layers.Conv2D(100, (5, self.EMB_DIM), padding='valid',
		                                   kernel_initializer='normal', activation='relu') # filters, kernel size
		self._max_pool_5 = layers.MaxPooling2D((self.MAX_LEN - 5 + 1, 1), strides=(1,1), padding='valid')

		self._fc_dense = layers.Flatten()
		self._dropout = layers.Dropout(0.5)
		# self._dense_out = layers.Dense(1, activation='sigmoid')
		self._dense_out = layers.Dense(2, activation='softmax')

	def call(self, x):

		emb_layer = self._embedding(x)
		emb_layer = self._reshape(emb_layer)

		# print(emb_layer.shape)
		# print(emb_layer)

		cnn_1 = self._cnn_filter_3(emb_layer)
		max_1 = self._max_pool_3(cnn_1)
		cnn_2 = self._cnn_filter_4(emb_layer)
		max_2 = self._max_pool_4(cnn_2)
		cnn_3 = self._cnn_filter_5(emb_layer)
		max_3 = self._max_pool_5(cnn_3)

		# print(max_1, max_2, max_3)

		concat = layers.concatenate([max_1, max_2, max_3])
		dense_fc = self._fc_dense(concat)
		drop_out = self._dropout(dense_fc)
		dense_out = self._dense_out(drop_out)

		return dense_out


class MLP(tf.keras.Model):

	def __init__(self, max_len, emb_dim, vocab_size):

		super(MLP, self).__init__()

		self.MAX_LEN = max_len
		self.EMB_DIM = emb_dim
		self.VOC_SIZE = vocab_size

		self._embedding = layers.Embedding(self.VOC_SIZE, self.EMB_DIM, input_length=self.MAX_LEN)

		self._pooling = keras.layers.GlobalAveragePooling1D()
		self._dense = keras.layers.Dense(16, activation='relu')
		# self._out = keras.layers.Dense(1, activation='sigmoid')
		self._out = keras.layers.Dense(2, activation='softmax')

	def call(self, x):

		emb_layer = self._embedding(x)

		avg_pooling_layer = self._pooling(emb_layer)
		dense = self._dense(avg_pooling_layer)
		out = self._out(dense)

		return out
