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
    test_data = Path.cwd() / 'data_in' / 'test.txt'

    with open(Path.cwd() / 'data_in' / 'vocab.pkl', mode='rb') as io:
        vocab = pickle.load(io)

    test = tf.data.TextLineDataset(str(test_data)).batch(batch_size=FLAGS.batch_size)

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

    test_loss = 0
    test_acc = 0
    test_len = 0
    tf.keras.backend.set_learning_phase(1)

    for step, val in enumerate(test):
        data, label = processing.token2idex(val)

        with tf.GradientTape() as tape:
            m = sen_cnn(data)
            mb_loss = loss_fn(label, m)
        grads = tape.gradient(target=mb_loss, sources=sen_cnn.trainable_variables)
        opt.apply_gradients(grads_and_vars=zip(grads, sen_cnn.trainable_variables))
        test_loss += mb_loss.numpy()

        prediction = tf.argmax(m, 1, name="predictions")
        test_acc += tf.reduce_sum(
            tf.cast(tf.equal(tf.cast(prediction, tf.int64), tf.cast(label, tf.int64)), tf.int64))
        test_len += batch_size

    else:
        test_loss /= (step + 1)
        test_acc += tf.reduce_sum(
            tf.cast(tf.equal(tf.cast(prediction, tf.int64), tf.cast(label, tf.int64)), tf.int64))
        test_len += batch_size

    tf.keras.backend.set_learning_phase(0)

    train_accu = test_acc / test_len * 100

    print("test_acc {}".format(test_acc))
    print("test_len {}".format(test_len))

    tqdm.write('epoch : {}, test_acc : {:.3f}%, test_loss : {:.3f}'.format(1, train_accu, test_loss))


if __name__ == '__main__':
    app.run(main)
