import tensorflow as tf
import pickle
from tensorflow import keras
from tensorflow.keras import layers

embedding_layer = layers.Embedding(1000, 32) # (samples, sequence_length) -> (samples, sequence_length, embedding_dim)

vocab_size = 10000
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

