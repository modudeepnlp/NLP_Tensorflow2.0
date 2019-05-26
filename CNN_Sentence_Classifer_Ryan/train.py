import os
import tensorflow as tf
import pickle
import pandas as pd

from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
# from model.data import Corpus
# from konlpy.tag import Mecab
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # For TEST

proj_dir = Path.cwd()
params = json.load((proj_dir / 'params' / 'config.json').open())
tr_data_path = params['filepath'].get('pre_tr')
tr_label_path = params['filepath'].get('pre_tr_label')
tr_configs_path = params['filepath'].get('pre_data_configs')

### Hyperparams ###
batch_size = params['training'].get('batch_size')
epochs = params['training'].get('epochs')
emb_dim = params['training'].get('emb_dim')
voc_size = params['training'].get('emb_dim')
max_len = params['padder'].get('length')
lr = params['training'].get('learning_rate')

tr_data = np.load(open(tr_data_path, 'rb'))
tr_label = np.load(open(tr_label_path, 'rb'))
tr_configs = json.load(open(tr_configs_path, encoding='utf-8'))
vocab_size = tr_configs['vocab_size'] + 1

tr_dataset = tf.data.Dataset.from_tensor_slices((tr_data, tr_label)).shuffle(len(tr_data))
tr_dataset = tr_dataset.batch(batch_size, drop_remainder=True)

class SimpleClassifier(tf.keras.Model):

	def __init__(self, max_len, emb_dim, vocab_size):

		super(SimpleClassifier, self).__init__()

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
		self._dense_out = layers.Dense(1, activation='sigmoid')
		# self._dense_out = layers.Dense(2, activation='softmax')

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

classifier = SimpleClassifier(max_len, emb_dim, vocab_size)
opt = tf.optimizers.Adam(learning_rate = lr)
loss_fn = tf.keras.metrics.binary_crossentropy

for epoch in tqdm(range(epochs), desc='epochs'):

	start = time.time()

	tr_loss = 0
	tf.keras.backend.set_learning_phase(1)

	for step, mb in tqdm(enumerate(tr_dataset), desc='steps'):
		x_mb, y_mb = mb

		with tf.GradientTape() as tape:
			mb_loss = tf.reduce_mean(loss_fn(y_mb, classifier(x_mb)))
		grads = tape.gradient(target=mb_loss, sources=classifier.trainable_variables)
		opt.apply_gradients(grads_and_vars=zip(grads, classifier.trainable_variables))

		# tr_loss += mb_loss.numpy()

		if step % 100 == 0:
			template = 'Epoch {} Batch {} Loss {:.4f} Time {:.4f}'
			print(template.format(epoch + 1, step, mb_loss, (time.time() - start)))

	else:
		tr_loss /= (step + 1)
		print(tr_loss)

	# tqdm.write('epoch : {}, tr_loss : {:.3f}, val_loss : {:.3f}'.format(epoch + 1, tr_loss))


