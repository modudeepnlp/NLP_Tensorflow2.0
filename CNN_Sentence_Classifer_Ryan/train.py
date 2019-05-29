import os
import tensorflow as tf
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
# from model.data import Corpus
# from konlpy.tag import Mecab
from model.net import CNN, MLP
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # For TEST

##Path
proj_dir = Path.cwd()
params = json.load((proj_dir / 'params' / 'config.json').open())
vocab_path = params['filepath'].get('vocab')

tr_data_path = params['filepath'].get('pre_tr')
tr_label_path = params['filepath'].get('pre_tr_label')
tr_configs_path = params['filepath'].get('pre_data_configs')

test_data_path = params['filepath'].get('pre_test')
test_label_path = params['filepath'].get('pre_test_label')

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

test_data = np.load(open(test_data_path, 'rb'))
test_label = np.load(open(test_label_path, 'rb'))

with open(vocab_path, mode='rb') as io:
	vocab = pickle.load(io)

tr_data, val_data, tr_label, val_label = train_test_split(tr_data, tr_label, test_size=0.2, random_state=777)

tr_dataset = tf.data.Dataset.from_tensor_slices((tr_data, tr_label)).shuffle(len(tr_data))
tr_dataset = tr_dataset.batch(batch_size, drop_remainder=True)
val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_label)).shuffle(len(val_data))
val_dataset = tr_dataset.batch(batch_size, drop_remainder=True)

test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_label)).batch(batch_size)

# classifier = MLP(max_len, emb_dim, vocab_size)
classifier = CNN(max_len, emb_dim, vocab_size)

# classifier.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
#                               min_delta=0,
#                               patience=3,
#                               verbose=0, mode='auto')
#
# history = classifier.fit(
# 	tr_dataset,
#     epochs=epochs,
# 	validation_data = (test_data, test_label),
# 	callbacks=[early_stopping])
#
# test_loss, test_acc = classifier.evaluate(test_dataset)

opt = tf.optimizers.Adam(learning_rate = lr)
# loss_fn = tf.keras.metrics.binary_crossentropy
loss_fn = tf.losses.SparseCategoricalCrossentropy()

# metrics
tr_loss_metric = tf.keras.metrics.Mean(name='train_loss')
tr_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
val_loss_metric = tf.keras.metrics.Mean(name='validation_loss')
val_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='validation_accuracy')

for epoch in tqdm(range(epochs), desc='epochs'):

	start = time.time()

	tr_loss = 0
	tf.keras.backend.set_learning_phase(1)

	for step, mb in tqdm(enumerate(tr_dataset), desc='steps'):
		x_mb, y_mb = mb

		with tf.GradientTape() as tape:
			mb_loss = loss_fn(y_mb, classifier(x_mb))
		grads = tape.gradient(target=mb_loss, sources=classifier.trainable_variables)
		opt.apply_gradients(grads_and_vars=zip(grads, classifier.trainable_variables))

		tr_loss_metric.update_state(mb_loss)
		tr_accuracy_metric(y_mb, classifier(x_mb))

		if step % 100 == 0:
			tr_mean_loss = tr_loss_metric.result()
			tr_mean_accuracy = tr_accuracy_metric.result()

			template = 'Epoch {} Batch {} Loss {:.4f} Acc {:.4f} Time {:.4f}'
			print(template.format(epoch + 1, step, tr_mean_loss, tr_mean_accuracy, (time.time() - start)))

		# tf.keras.backend.set_learning_phase(0)
		#
		# for _, mb in tqdm(enumerate(val_dataset), desc='steps'):
		# 	x_mb, y_mb = mb
		# 	mb_loss = loss_fn(y_mb, classifier(x_mb))
		# 	val_loss_metric.update_state(mb_loss)
		# 	val_accuracy_metric.update_state(y_mb, classifier(x_mb))
		#
		# val_mean_loss = val_loss_metric.result()
		# val_mean_accuracy = val_accuracy_metric.result()
		#
		# val_template = 'Loss {:.4f}, Acc {:.4f}'
		# print(val_template.format(val_mean_loss, val_mean_accuracy))

		# tqdm.write('epoch : {}, tr_loss : {:.3f}, val_loss : {:.3f}'.format(epoch + 1, tr_loss))


