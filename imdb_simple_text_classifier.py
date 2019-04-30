import tensorflow as tf
import pickle
from tensorflow import keras
from tensorflow.keras import layers

embedding_layer = layers.Embedding(1000, 32) # (samples, sequence_length) -> (samples, sequence_length, embedding_dim)

vocab_size = 10000
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

# integers에서 words로 변환
word_index = imdb.get_word_index()

# 첫 값들을 넣기 위하여 +3을 해준다
word_index = {k: (v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
	return ' '.join([reverse_word_index.get(i, '?') for i in text])

decode_review(train_data[0])

# Pad length를 통해 추가로 길이를 맞추어 주자
maxlen = 500
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=maxlen)

test_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=maxlen)

"""
1. Embedding Layer: integer를 embedding으로 변환, (batch, sequence, embedding) 
2. GlobalAveragePooling1D: 
3. fully-connected (Dense) with 16 hidden units
4. densely connected with single output node + sigmoid activation prob. the review is positive or not
"""

embedding_dim = 16

model = keras.Sequential([
  layers.Embedding(vocab_size, embedding_dim, input_length=maxlen),
  layers.GlobalAveragePooling1D(),
  layers.Dense(16, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_data,
    train_labels,
    epochs=30,
    batch_size=512,
    validation_split=0.2)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
matplotlib.get_backend()

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12,9))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim((0.5,1))

# plt.show()
plt.savefig('imdb_simple_text_classifier.png')
