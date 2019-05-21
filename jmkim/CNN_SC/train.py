import tensorflow as tf
import pickle
import pandas as pd
import numpy as np

from pathlib import Path
from absl import app
from konlpy.tag import Mecab
from model.data import Corpus
from tqdm import tqdm

from model.net import SenCNN
from configs import FLAGS


def main(argv):
	train_data = Path.cwd() / 'data_in' / 'train.txt'
	val_data = Path.cwd() / 'data_in' / 'val.txt'

	with open(Path.cwd() / 'data_in' / 'vocab.pkl', mode='rb') as io:
		vocab = pickle.load(io)

	train = tf.data.TextLineDataset(str(train_data)).shuffle(buffer_size=1000).batch(batch_size=FLAGS.batch_size)
	eval = tf.data.TextLineDataset(str(val_data)).shuffle(buffer_size=1000).batch(batch_size=FLAGS.batch_size)

	tokenized = Mecab()
	processing = Corpus(vocab=vocab, tokenizer=tokenized)

	# init params
	classes = FLAGS.classes
	max_length = FLAGS.length
	epochs = FLAGS.epochs
	batch_size = FLAGS.batch_size
	learning_rate = FLAGS.learning_rate

	# create model
	sen_cnn = SenCNN(vocab=vocab, max_len=max_length, dim=batch_size, classes=classes)

	# create optimizer & loss_fn
	opt = tf.optimizers.Adam(learning_rate=learning_rate)
	loss_fn = tf.losses.SparseCategoricalCrossentropy()

	train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
	train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
	val_loss_metric = tf.keras.metrics.Mean(name='train_loss')
	val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

	# training
	for epoch in tqdm(range(epochs)):

		tr_loss = 0
		tf.keras.backend.set_learning_phase(1)
		train_loss_metric.reset_states()
		train_acc_metric.reset_states()

		for step, val in enumerate(train):
			data, label = processing.token2idex(val)

			'''
			이 밑은 좀 공부 할것.
			loss 관련
			'''
			with tf.GradientTape() as tape:
				logits = sen_cnn(data)
				pred_loss = loss_fn(label, logits)
				total_loss = pred_loss

			grads = tape.gradient(target=total_loss, sources=sen_cnn.trainable_variables)
			opt.apply_gradients(grads_and_vars=zip(grads, sen_cnn.trainable_variables))

			train_loss_metric.update_state(total_loss)
			train_acc_metric.update_state(label, logits)

		tr_loss = train_loss_metric.result()
				  #/ (step + 1)

		tf.keras.backend.set_learning_phase(0)

		val_loss = 0
		for step, val in enumerate(eval):
			data, label = processing.token2idex((val))
			logits = sen_cnn(data)
			mb_loss = loss_fn(label, logits)
			val_loss_metric.update_state(mb_loss)
			val_acc_metric.update_state(label, logits)

		val_loss = val_loss_metric.result()
				   #/ (step + 1)

		print("tr_acc {}%".format(train_acc_metric.result() * 100))

		tqdm.write('epoch : {}, tr_acc : {:.3f}%, tr_loss : {:.3f}, val_acc : {:.3f}%, val_loss : {:.3f}'.format(epoch + 1, train_acc_metric.result() * 100, tr_loss, val_acc_metric.result() * 100, val_loss))


if __name__ == '__main__':
	app.run(main)
