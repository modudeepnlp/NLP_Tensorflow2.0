import pandas as pd

import tensorflow as tf
from gluonnlp.data import PadSequence
from gluonnlp import Vocab
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Tuple

from configs import FLAGS


class Corpus():
    def __init__(self, vocab, tokenizer):
        self._vocab = vocab
        self._toknizer = tokenizer

    def token2idex(self, item):
        data, label = tf.io.decode_csv(item, record_defaults=[[''], [0]], field_delim='\t')
        data = [[self._vocab.token_to_idx[token] for token in self._toknizer.morphs(sen.numpy().decode('utf-8'))] for
                sen in item]
        data = pad_sequences(data, maxlen=FLAGS.length, value=self._vocab.token_to_idx['<pad>'], padding='post', truncating='post')
        #data = PadSequence(length=FLAGS.length, pad_val=self._vocab.token_to_idx['<pad>'])(data)
        data = tf.convert_to_tensor(data, dtype=tf.int32)
        label = tf.reshape(label, (item.get_shape()[0], ))
        return data, label
