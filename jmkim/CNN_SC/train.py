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

    # training
    for epoch in tqdm(range(epochs)):

        tr_loss = 0
        tr_acc = 0
        tr_len = 0
        tf.keras.backend.set_learning_phase(1)

        for step, val in enumerate(train):
            data, label = processing.token2idex(val)

            '''
            이 밑은 좀 공부 할것.
            loss 관련
            '''
            with tf.GradientTape() as tape:
                m = sen_cnn(data)
                mb_loss = loss_fn(label, m)
            grads = tape.gradient(target=mb_loss, sources=sen_cnn.trainable_variables)
            opt.apply_gradients(grads_and_vars=zip(grads, sen_cnn.trainable_variables))
            tr_loss += mb_loss.numpy()

            prediction = tf.argmax(m, 1, name="predictions")
            tr_acc += tf.reduce_sum(
                tf.cast(tf.equal(tf.cast(prediction, tf.int64), tf.cast(label, tf.int64)), tf.int64))
            tr_len += batch_size

        else:
            '''
            근데 이 else는 누구랑 세트인가??
            if가 없는디.. python 문법에 이런게 있구나.
            '''
            tr_loss /= (step + 1)
            tr_acc += tf.reduce_sum(
                tf.cast(tf.equal(tf.cast(prediction, tf.int64), tf.cast(label, tf.int64)), tf.int64))
            tr_len += batch_size

        tf.keras.backend.set_learning_phase(0)

        val_loss = 0
        val_acc = 0
        val_len = 0
        for step, val in enumerate(eval):
            data, label = processing.token2idex((val))
            m = sen_cnn(data)
            mb_loss = loss_fn(label, m)
            val_loss += mb_loss.numpy()
            prediction = tf.argmax(m, 1, name="predictions")
            val_acc += tf.reduce_sum(tf.cast(tf.equal(tf.cast(prediction,tf.int64), tf.cast(label, tf.int64)), tf.int64))
            val_len += batch_size
        else:
            val_loss /= (step + 1)
            val_acc += tf.reduce_sum(tf.cast(tf.equal(tf.cast(prediction,tf.int64), tf.cast(label, tf.int64)), tf.int64))
            val_len += batch_size

        train_accu = tr_acc / tr_len * 100
        val_accu = val_acc / val_len * 100

        print("tr_acc {}".format(tr_acc))
        print("tr_len {}".format(tr_len))
        print("val_acc {}".format(val_acc))
        print("val_len {}".format(val_len))

        tqdm.write('epoch : {}, tr_acc : {:.3f}%, tr_loss : {:.3f}, val_acc : {:.3f}%, val_loss : {:.3f}'.format(epoch + 1, train_accu, tr_loss, val_accu, val_loss))


if __name__ == '__main__':
    app.run(main)
