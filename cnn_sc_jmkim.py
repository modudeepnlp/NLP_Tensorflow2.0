import os
import tensorflow as tf
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

def cnn():
    train_input_data = np.load(open('./' + FLAGS.input_data + '/' + FLAGS.train_npy, 'rb'))
    train_label_data = np.load(open('./' + FLAGS.input_data + '/' + FLAGS.train_label_npy, 'rb'))
    prepro_configs = json.load(open('./' + FLAGS.input_data + '/dic_config.json', 'r'))

    input_train, input_eval, label_train, label_eval = train_test_split(train_input_data, train_label_data,
                                                                        test_size=0.2,
                                                                        random_state=777)
    cnn3 = tf.keras.models.Sequential()
    cnn3.add(tf.keras.layers.Embedding(prepro_configs['vocab_size'], 128))
    cnn3.add(tf.keras.layers.Dropout(0.5))
    cnn3.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding="VALID"))
    cnn3.add(tf.keras.layers.GlobalMaxPool1D())

    cnn4 = tf.keras.models.Sequential()
    cnn4.add(tf.keras.layers.Embedding(prepro_configs['vocab_size'], 128))
    cnn4.add(tf.keras.layers.Dropout(0.5))
    cnn4.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding="VALID"))
    cnn4.add(tf.keras.layers.GlobalMaxPool1D())

    cnn5 = tf.keras.models.Sequential()
    cnn5.add(tf.keras.layers.Embedding(prepro_configs['vocab_size'], 128))
    cnn5.add(tf.keras.layers.Dropout(0.5))
    cnn5.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding="VALID"))
    cnn5.add(tf.keras.layers.GlobalMaxPool1D())

    list = [cnn3, cnn4, cnn5]
    model = tf.keras.layers.concatenate(list) # list가 2보다 작다는 에러만 나옴.
    model.add(tf.keras.layers.Dense(250, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    history = model.fit(input_train, label_train,
                        epochs=FLAGS.epochs,
                        batch_size=FLAGS.batch_size,
                        validation_data=(input_eval, label_eval),
                        verbose=1)

    test_input_data = np.load(open('./' + FLAGS.input_data + '/' + FLAGS.test_npy, 'rb'))
    test_label_data = np.load(open('./' + FLAGS.input_data + '/' + FLAGS.test_label_npy, 'rb'))
    results = model.evaluate(test_input_data, test_label_data)
    print(results)

def main(argv):
    basic()
    cnn()

if __name__ == '__main__':
    app.run(main)