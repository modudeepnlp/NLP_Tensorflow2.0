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
        token_idx = self.split_transform(data)
        label = tf.convert_to_tensor(label)
        print(label)
        return token_idx, label
