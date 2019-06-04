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
    sen_cnn = SenCNN(vocab=vocab, classes=classes)

    # create optimizer & loss_fn
    opt = tf.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.losses.SparseCategoricalCrossentropy()

    test_loss_metric = tf.keras.metrics.Mean(name='val_loss')
    test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=sen_cnn)
    manager = tf.train.CheckpointManager(ckpt, './data_out/tf_ckpts', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    tf.keras.backend.set_learning_phase(0)
    test_loss_metric.reset_states()
    test_acc_metric.reset_states()

    for step, val in enumerate(test):
        data, label = processing.token2idex(val)
        logits = sen_cnn(data)
        val_loss = loss_fn(label, logits)
        # val_loss += mb_loss.numpy()
        test_loss_metric.update_state(val_loss)
        test_acc_metric.update_state(label, logits)

    test_loss = test_loss_metric.result()

    tqdm.write(
        'epoch : {}, tr_acc : {:.3f}%, tr_loss : {:.3f}, '.format(1, test_acc_metric.result() * 100, test_loss))

if __name__ == '__main__':
    app.run(main)
