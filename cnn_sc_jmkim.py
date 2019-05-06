import os
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import json

from datetime import datetime
from sklearn.model_selection import train_test_split
from absl import app
from configs import FLAGS


def basic():
    train_input_data = np.load(open('./' + FLAGS.input_data + '/' + FLAGS.train_npy, 'rb'))
    train_label_data = np.load(open('./' + FLAGS.input_data + '/' + FLAGS.train_label_npy, 'rb'))
    prepro_configs = json.load(open('./' + FLAGS.input_data + '/dic_config.json', 'r'))

    input_train, input_eval, label_train, label_eval = train_test_split(train_input_data, train_label_data,
                                                                        test_size=0.2,
                                                                        random_state=777)
    basic = tf.keras.Sequential()
    basic.add(tf.keras.layers.Embedding(prepro_configs['vocab_size'], FLAGS.batch_size, input_shape=(None,)))
    basic.add(tf.keras.layers.Dropout(0.2))
    basic.add(tf.keras.layers.GlobalAveragePooling1D())
    basic.add(tf.keras.layers.Dropout(0.2))
    basic.add(tf.keras.layers.Dense(FLAGS.batch_size, activation=tf.nn.relu))
    basic.add(tf.keras.layers.Dropout(0.2))
    basic.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    basic.summary()

    basic.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    history = basic.fit(input_train, label_train,
                        epochs=FLAGS.epochs,
                        batch_size=FLAGS.batch_size,
                        validation_data=(input_eval, label_eval),
                        verbose=1)

    test_input_data = np.load(open('./' + FLAGS.input_data + '/' + FLAGS.test_npy, 'rb'))
    test_label_data = np.load(open('./' + FLAGS.input_data + '/' + FLAGS.test_label_npy, 'rb'))
    results = basic.evaluate(test_input_data, test_label_data)
    print('test')
    print(results)


class CNN(tf.keras.Model):
    def __init__(self, MAX_LEN, EMB, VOCAB_SIZE):
        super(CNN, self).__init__()

        self._embedding = tf.keras.layers.Embedding(VOCAB_SIZE, EMB, input_length=MAX_LEN)
        self._reshape = tf.keras.layers.Reshape((MAX_LEN, EMB, 1))

        self._cnn3 = tf.keras.layers.Conv2D(100, kernel_size=(3, EMB), padding='valid',
                                            kernel_initializer='normal', activation='relu')
        self._maxpool3 = tf.keras.layers.MaxPooling2D((MAX_LEN - 3 + 1, 1), strides=(1, 1), padding='valid')
        self._cnn4 = tf.keras.layers.Conv2D(100, kernel_size=(4, EMB), padding='valid',
                                            kernel_initializer='normal', activation='relu')
        self._maxpool4 = tf.keras.layers.MaxPooling2D((MAX_LEN - 4 + 1, 1), strides=(1, 1), padding='valid')
        self._cnn5 = tf.keras.layers.Conv2D(100, kernel_size=(5, EMB), padding='valid',
                                            kernel_initializer='normal', activation='relu')
        self._maxpool5 = tf.keras.layers.MaxPooling2D((MAX_LEN - 5 + 1, 1), strides=(1, 1), padding='valid')

        self._fc_dense = tf.keras.layers.Flatten()
        self._dropout = tf.keras.layers.Dropout(0.5)
        self._dense_out = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x):
        embedding = self._embedding(x)
        embedding = self._reshape(embedding)

        cnn3 = self._cnn3(embedding)
        maxpool3 = self._maxpool3(cnn3)
        cnn4 = self._cnn4(embedding)
        maxpool4 = self._maxpool4(cnn4)
        cnn5 = self._cnn5(embedding)
        maxpool5 = self._maxpool5(cnn5)

        concat = tf.keras.layers.concatenate([maxpool3, maxpool4, maxpool5])
        dense = self._fc_dense(concat)
        dropout = self._dropout(dense)
        dense_out = self._dense_out(dropout)

        return dense_out


def cnn():
    train_input_data = np.load(open('./' + FLAGS.input_data + '/' + FLAGS.train_npy, 'rb'))
    train_label_data = np.load(open('./' + FLAGS.input_data + '/' + FLAGS.train_label_npy, 'rb'))
    prepro_configs = json.load(open('./' + FLAGS.input_data + '/dic_config.json', 'r'))

    train_input_data = tf.keras.preprocessing.sequence.pad_sequences(train_input_data,
                                                                     value=1,
                                                                     padding='post',
                                                                     maxlen=FLAGS.length)

    cnn_sc = CNN(FLAGS.length, 128, prepro_configs['vocab_size'])
    cnn_sc.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    cnn_sc.fit(train_input_data, train_label_data,
               epochs=FLAGS.epochs,
               batch_size=FLAGS.batch_size,
               validation_split=0.2,
               verbose=1)

    test_input_data = np.load(open('./' + FLAGS.input_data + '/' + FLAGS.test_npy, 'rb'))
    test_label_data = np.load(open('./' + FLAGS.input_data + '/' + FLAGS.test_label_npy, 'rb'))

    test_input_data = tf.keras.preprocessing.sequence.pad_sequences(test_input_data,
                                                                    value=1,
                                                                    padding='post',
                                                                    maxlen=FLAGS.length)

    results = cnn_sc.evaluate(test_input_data, test_label_data)
    print(results)


def main(argv):
    # basic()
    cnn()


if __name__ == '__main__':
    app.run(main)
