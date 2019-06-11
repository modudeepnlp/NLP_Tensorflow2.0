import tensorflow as tf


class Corpus():
    def __init__(self, vocab, split_fn, pad_fn):
        self._vocab = vocab
        self._split = split_fn
        self._padding = pad_fn

    def split(self, string):
        return self._split.sentenceSplit(string)

    def transform(self, list_token):
        list_to_idx = [self._vocab.to_indices(token) for token in list_token]
        list_to_idx = [self._padding(token) if self._padding else token for token in list_to_idx]

        return list_to_idx

    def split_transform(self, string):
        return self.transform(self.split(string))

    def token2idex(self, item):
        data, label = tf.io.decode_csv(item, record_defaults=[[''], [0]], field_delim='\t')
        data = [sen.numpy().decode('utf-8') for sen in data]
        data = self.split_transform(data)
        token_idx = tf.convert_to_tensor(data)
        #label = [float(v) for v in label]
        label = tf.reshape(label, (item.get_shape()[0],))
        #label = tf.convert_to_tensor([float(v) for v in label])
        return token_idx, label
