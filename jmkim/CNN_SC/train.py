import tensorflow as tf
import pickle
import pandas as pd

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
    sen_cnn = SenCNN(vocab=vocab, max_len=max_length, dim=batch_size, classes=2)

    # create optimizer & loss_fn
    opt = tf.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.losses.SparseCategoricalCrossentropy()

    # training
    for epoch in tqdm(range(epochs)):

        tr_loss = 0
        tf.keras.backend.set_learning_phase(1)

        for step, val in tqdm(enumerate(train)):
            data, label = processing.token2idex(val)

            '''
            이 밑은 좀 공부 할것.
            loss
            '''
            with tf.GradientTape() as tape:
                mb_loss = loss_fn(label, sen_cnn(data))
            grads = tape.gradient(target=mb_loss, sources=sen_cnn.trainable_variables)
            opt.apply_gradients(grads_and_vars=zip(grads, sen_cnn.trainable_variables))
            tr_loss += mb_loss.numpy()
        else:
            '''
            근데 이 else는 누구랑 세트인가??
            if가 없는디.. python 문법에 이런게 있구나.
            '''
            tr_loss /= (step + 1)

        tf.keras.backend.set_learning_phase(0)
        val_loss = 0
        for step, val in tqdm(enumerate(eval), desc='steps'):
            data, label = processing.token2idex((val))
            mb_loss = loss_fn(label, sen_cnn(data))
            val_loss += mb_loss.numpy()
        else:
            val_loss /= (step + 1)

        tqdm.write('epoch : {}, tr_loss : {:.3f}, val_loss : {:.3f}'.format(epoch + 1, tr_loss, val_loss))

if __name__ == '__main__':
    app.run(main)
