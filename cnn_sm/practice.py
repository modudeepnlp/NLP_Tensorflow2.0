from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
from cnn_sm.model.ops import MultiChannelEmbedding, ConvolutionLayer, MaxPooling
from model.net import SenCNN
from gluonnlp import Vocab
import json
from konlpy.tag import Mecab
import pickle
from model.utils import PreProcessor
from pathlib import Path
import pandas as pd
from tqdm import tqdm

class SmCnn(tf.keras.Model):

    def __init__(self, num_classes=2, batch_size=128, learning_rate=1e-3, epochs=10,
                 filter_size=[3, 4, 5], drop_rate=0.5, vocab=Vocab):
        super(SmCnn, self).__init__()
        self._embedding = MultiChannelEmbedding(vocab)
        self._convolution = ConvolutionLayer(300)
        self._pooling = MaxPooling()
        self._dropout = drop_rate
        self._batch_size = batch_size
        self._fc = layers.Dense(units=num_classes)
        self._epochs = epochs
        self._learning_rate = learning_rate

    def call(self, x):
        fmap = self._embedding(x)
        fmap = self._convolution(fmap)
        feature = self._pooling(fmap)
        feature = self._dropout(feature)
        score = self._fc(feature)
        return score

    def create_dataset(self, filepath, batch_size, shuffle=True, drop_remainder=True):
        ds = tf.data.TextLineDataset(filepath)
        if shuffle:
            ds = ds.shuffle(buffer_size=1000)

        ds = ds.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return ds

    def main(self):
        batch_size = self._batch_size
        tr_filepath = 'data/train.txt'
        val_filepath = 'data/val.txt'
        tr_ds = SmCnn.create_dataset(self, tr_filepath, batch_size, shuffle=True)
        val_ds = SmCnn.create_dataset(self, val_filepath, batch_size, shuffle=False)

        vocab = pd.read_pickle('data/vocab.pkl')
        pre_processor = PreProcessor(vocab=vocab, tokenizer=Mecab())

        # create model
        model = SenCNN(num_classes=2, vocab=vocab)

        # create optimizer & loss_fn
        epochs = self._epochs
        learning_rate = self._learning_rate
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()


        # training
        for epoch in tqdm(range(epochs), desc='steps'):
            tr_loss = 0
            tf.keras.backend.set_learning_phase(1)

            for step, mb in tqdm(enumerate(tr_ds), desc='steps'):
                x_mb, y_mb = pre_processor.convert2idx(mb)
                with tf.GradientTape() as tape:
                    mb_loss = loss_fn(y_mb, model(x_mb))
                grads = tape.gradient(target=mb_loss, sources=model.trainable_variables)
                opt.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
                tr_loss += mb_loss.numpy()
            else:
                tr_loss /= (step + 1)

            tf.keras.backend.set_learning_phase(0)
            val_loss = 0
            for step, mb in tqdm(enumerate(val_ds), desc='steps'):
                x_mb, y_mb = pre_processor.convert2idx(mb)
                mb_loss = loss_fn(y_mb, model(x_mb))
                val_loss += mb_loss.numpy()
            else:
                val_loss /= (step + 1)

            tqdm.write('epoch : {}, tr_loss : {:.3f}, val_loss : {:.3f}'.format(epoch + 1, tr_loss, val_loss))

if __name__ == '__main__':
    cnn_model = SmCnn()
    cnn_model.main()