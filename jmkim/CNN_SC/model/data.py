import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from configs import FLAGS


class Corpus():
    def __init__(self, vocab, tokenizer):
        self._vocab = vocab
        self._toknizer = tokenizer

    def token2idex(self, item):
        data, label = tf.io.decode_csv(item, record_defaults=[[''], [0]], field_delim='\t')
        data = [self._toknizer.morphs(sen.numpy().decode('utf-8')) for sen in data]
        data = [[self._vocab.token_to_idx[token] for token in sen] for sen in data]
        data = pad_sequences(data, maxlen=FLAGS.length, value=0,
                             padding='post', truncating='post')

        data = tf.convert_to_tensor(data, dtype=tf.int32)
        label = tf.reshape(label, (item.get_shape()[0], ))
        return data, label
