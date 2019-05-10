import tensorflow as tf
import pickle
import pandas as pd

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from pathlib import Path
from absl import app
from konlpy.tag import Mecab
from model.data import Corpus
from model.net import SenCNN
from configs import FLAGS



def main(argv):
    train_data = Path.cwd()  / 'data_in' / 'train.txt'
    val_data = Path.cwd()  / 'data_in' / 'val.txt'

    with open(Path.cwd()  / 'data_in' / 'vocab.pkl', mode='rb') as io:
        vocab = pickle.load(io)

    # init params
    classes = FLAGS.classes
    max_length = FLAGS.length
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    learning_rate = FLAGS.learning_rate

    # create model
    sen_cnn = SenCNN(vocab=vocab, max_len=max_length, dim=batch_size)
    print(sen_cnn)

    # create data
    tokenizer = Mecab()

    #padder
    #train_data
    #train_data_load
    #val_data
    #val_data_load


    # training


if __name__ == '__main__':
    app.run(main)
