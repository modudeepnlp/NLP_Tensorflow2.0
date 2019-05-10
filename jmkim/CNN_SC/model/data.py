import pandas as pd

import tensorflow as tf
from gluonnlp.data import PadSequence
from gluonnlp import Vocab
from typing import Tuple

from configs import FLAGS


class Corpus():
    def __init__(self, filepath, vocab, tokenizer, padder):

        self._corpus = pd.read_csv(filepath, sep='\t').loc[:, ['document', 'label']]
        self._vocab = vocab
        self._toknizer = tokenizer
        self._padder = padder

    def __len__(self):
        return len(self._corpus)

    def __getitem__(self, item):
        tokenized = self._toknizer.morphs(self._corpus.iloc[item]['document'])
        tokenized2indices = tf.convert_to_tensor(self._padder([self._vocab.token_to_idx[token] for token in tokenized]))
        label = tf.convert_to_tensor(self._corpus.iloc[item]['label'])
        return tokenized2indices, label


