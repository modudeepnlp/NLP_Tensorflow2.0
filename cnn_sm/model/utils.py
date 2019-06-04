import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt
from gluonnlp import Vocab
from typing import Tuple


class PreProcessor:
    def __init__(self, vocab: Vocab, tokenizer: Okt, pad_idx: int = 0) -> None:
        '''

        :param vocab: gluonnlp.Vocab
        :param tokenizer: Okt
        :param pad_idx: the idx of padding token. Default: 0
        :param pad_length(int): padding length, Default: 70
        '''
        self._vocab = vocab
        self._tokenizer = tokenizer
        self._pad_idx = pad_idx

    def convert2idx(self, record: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        data, label = tf.io.decode_csv(record, record_defaults=[[''], [0]], field_delim='\t')
        data = [self._tokenizer.morphs(sen.numpy().decode('utf-8')) for sen in data]
        data = [[self._vocab.token_to_idx[token] for token in sen] for sen in data]

        return data, label

    def pad_sequences(self, data, pad_length):
        data = pad_sequences(data, maxlen=pad_length, value=self._pad_idx, padding='post', truncating='post')

        return data

    def convert_to_tensor(self, data, label):
        data = tf.convert_to_tensor(data, dtype=tf.int32)
        label = tf.reshape(label, (data.get_shape()[0],))

        return data, label
