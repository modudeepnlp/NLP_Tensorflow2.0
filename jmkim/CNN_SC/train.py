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
    sen_cnn = SenCNN(vocab=vocab, max_len=max_length, dim=batch_size)
    print(sen_cnn)

    # training
    for epoch in tqdm(range(epochs)):
        for step, val in tqdm(enumerate(train)):
            data = processing.token2idex(val)
            s = sen_cnn(data)
            print(s)
            break


if __name__ == '__main__':
    app.run(main)
