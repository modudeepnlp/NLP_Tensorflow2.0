"""
Quora: 363861/363861 [==============================] - 79s 216us/sample - loss: 0.0927 - acc: 0.8756 - val_loss: 0.1586 - val_acc: 0.7846
6200/6200 [==============================] - 10s 2ms/sample - loss: 0.2157 - acc: 0.6853 - val_loss: 0.1997 - val_acc: 0.7282
"""

import os
import sys

from time import time
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# from tensorflow.python.keras.models import Model, Sequential
# from tensorflow.python.keras.layers import Input, Embedding, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPool1D, Dense, Dropout, Bidirectional

from model.utils import make_w2v_embeddings
from model.utils import split_and_zero_padding

# from lib.utils import ManDist

from pathlib import Path

proj_dir = Path.cwd()
print(proj_dir)

os.environ["CUDA_VISIBLE_DEVICES"]="7" #For TEST

# question_train_quora = '../data/quora-question-pairs_eng/train.csv'
question_train_kr = '../data/question-pair-kr/kor_pair_train.csv'
question_test_kr = '../data/question-pair-kr/kor_pair_test.csv'

train_df = pd.read_csv(question_train_kr)
test_df = pd.read_csv(question_test_kr)

for q in ['question1', 'question2']:
    train_df[q + '_n'] = train_df[q]
# Make word2vec embeddings
MAXLEN = 50
EMB_DIM = 300
use_w2v = False

train_df, embeddings = make_w2v_embeddings(train_df, embedding_dim=EMB_DIM, empty_w2v=not use_w2v)

# Model variables
gpus = 1
batch_size = 64 * gpus
n_epoch = 100
n_hidden = 50

VOC_SIZE = len(embeddings)

# Split to train validation
validation_size = int(len(train_df) * 0.1)
training_size = len(train_df) - validation_size

X = train_df[['question1_n', 'question2_n']]
Y = train_df['is_duplicate']

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

X_train = split_and_zero_padding(X_train, MAXLEN)
X_validation = split_and_zero_padding(X_validation, MAXLEN)

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)


class SelfAttention(tf.keras.Model):

    def __init__(self, da, r):

        super(SelfAttention, self).__init__()
        self._ws1 = layers.Dense(da, use_bias=False)
        self._ws2 = layers.Dense(r, use_bias=False)

    def call(self, x_hidden):

        att_layer = self._ws2(tf.nn.tanh(self._ws1(x_hidden)))
        att_layer = tf.nn.softmax(att_layer)

        return att_layer

class SAN(tf.keras.Model):

    def __init__(self, max_len, emb_dim, n_hidden, a, d, vocab_size):
        """
        :param max_len:
        :param emb_dim:
        :param n_hidden:
        :param a:
        :param d:
        :param vocab_size:
        """

        super(SAN, self).__init__()

        self.MAX_LEN = max_len
        self.EMB_DIM = emb_dim
        self.VOC_SIZE = vocab_size

        self._embedding = layers.Embedding(self.VOC_SIZE, self.EMB_DIM, input_length=self.MAX_LEN)
        self._bilstm = layers.Bidirectional(layers.LSTM(150, return_sequences=False), input_shape=(50, 300), name='shared_internal_layer')
        # self._bilstm_last = layers.Bidirectional(layers.LSTM(n_hidden, return_sequences=False))
        self._attn = SelfAttention(a, d)

        self._top_dense = layers.Dense(units=100, activation='relu')
        self._last_dense = layers.Dense(units=1, activation='sigmoid')
        # self._last_dense = layers.Dense(units=2, activation='softmax')
        # self._last_dense = layers.Dense(units=2, activation='softmax')

    def call(self, x):

        q1_emb_layer = self._embedding(x[0])
        q2_emb_layer = self._embedding(x[1])

        q1_lstm = self._bilstm(q1_emb_layer)
        # q1_lstm = self._bilstm_last(q1_lstm)
        q1_att = self._attn(q1_lstm)

        q2_lstm = self._bilstm(q2_emb_layer)
        # q2_lstm = self._bilstm(q2_lstm)

        print("q2_lstm pass", q2_lstm)

        q2_att = self._attn(q2_lstm)

        print("att pass", q2_att)

        Fr = q1_att * q2_att

        print("fr pass", Fr)

        Fr = self._top_dense(Fr)
        Fr = self._top_dense(Fr)

        last = self._last_dense(Fr)

        return last

sent_sim = SAN(max_len=MAXLEN, emb_dim=EMB_DIM, n_hidden=n_hidden, a=30, d=100, vocab_size=VOC_SIZE)

if gpus >= 2:
    # `multi_gpu_model()` is a so quite buggy. it breaks the saved model.
    sent_sim = tf.keras.utils.multi_gpu_model(sent_sim, gpus=gpus)

sent_sim.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=0, mode='auto')

# filepath = 'weights.h5'
# checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,
#                                                 monitor='val_loss',
#                                                 verbose=1,
#                                                 save_best_only=True,
#                                                 mode='min')

# Start trainings
training_start_time = time()

history = sent_sim.fit(
    [X_train['left'], X_train['right']], Y_train,
    batch_size=batch_size, epochs=n_epoch,
    callbacks=[early_stopping],
    validation_data=([X_validation['left'], X_validation['right']], Y_validation)
)

training_end_time = time()
print("Training time finished.\n%d epochs in %12.2f" % (n_epoch,
                                                        training_end_time - training_start_time))

def plot_graphs(history, string):
	plt.plot(history.history[string])
	plt.plot(history.history['val_'+string])
	plt.xlabel("Epochs")
	plt.ylabel(string)
	plt.legend([string, 'val_'+string])
	plt.savefig(string + '_ma_lstm.png')

# plot_graphs(history, 'accuracy')
# plot_graphs(history, 'loss')

sent_sim.save_weights('../data/san', save_format='tf')

