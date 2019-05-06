import tensorflow as tf
from tensorflow.keras import layers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_graphs(history, string):
	plt.plot(history.history[string])
	plt.plot(history.history['val_'+string])
	plt.xlabel("Epochs")
	plt.ylabel(string)
	plt.legend([string, 'val_'+string])
	plt.savefig('cnn_classifier.png')

# Pad length를 통해 추가로 길이를 맞추어 주자
# MAXLEN = 500
MAXLEN = 500
EMB_DIM = 128
VOC_SIZE = 10000
use_w2v = True

# 데이터셋 다운 받기
imdb = tf.keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOC_SIZE)

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

train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=MAXLEN)

test_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=MAXLEN)


class SimpleClassifier(tf.keras.Model):

	def __init__(self, max_len, emb_dim, vocab_size):

		super(SimpleClassifier, self).__init__()

		self.MAX_LEN = max_len
		self.EMB_DIM = emb_dim
		self.VOC_SIZE = vocab_size

		self._embedding = layers.Embedding(self.VOC_SIZE, self.EMB_DIM, input_length=self.MAX_LEN)
		self._reshape = layers.Reshape((self.MAX_LEN, self.EMB_DIM, 1))

		self._cnn_filter_3 = layers.Conv2D(100, kernel_size=(3, self.EMB_DIM), padding='valid',
		                                   kernel_initializer='normal', activation='relu') # filters, kernel size
		self._max_pool_3 = layers.MaxPooling2D((self.MAX_LEN - 3 + 1, 1), strides=(1,1), padding='valid')

		self._cnn_filter_4 = layers.Conv2D(100, (4, self.EMB_DIM), padding='valid',
		                                   kernel_initializer='normal', activation='relu') # filters, kernel size
		self._max_pool_4 = layers.MaxPooling2D((self.MAX_LEN - 4 + 1, 1), strides=(1,1), padding='valid')

		self._cnn_filter_5 = layers.Conv2D(100, (5, self.EMB_DIM), padding='valid',
		                                   kernel_initializer='normal', activation='relu') # filters, kernel size
		self._max_pool_5 = layers.MaxPooling2D((self.MAX_LEN - 5 + 1, 1), strides=(1,1), padding='valid')

		self._fc_dense = layers.Flatten()
		self._dropout = layers.Dropout(0.5)
		self._dense_out = layers.Dense(1, activation='sigmoid')

	def call(self, x):

		emb_layer = self._embedding(x)
		emb_layer = self._reshape(emb_layer)

		cnn_1 = self._cnn_filter_3(emb_layer)
		max_1 = self._max_pool_3(cnn_1)
		cnn_2 = self._cnn_filter_4(emb_layer)
		max_2 = self._max_pool_4(cnn_2)
		cnn_3 = self._cnn_filter_5(emb_layer)
		max_3 = self._max_pool_5(cnn_3)

		concat = layers.concatenate([max_1, max_2, max_3])
		dense_fc = self._fc_dense(concat)
		drop_out = self._dropout(dense_fc)
		dense_out = self._dense_out(drop_out)

		return dense_out

classifier = SimpleClassifier(MAXLEN, EMB_DIM, VOC_SIZE)

classifier.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')

history = classifier.fit(
    train_data,
    train_labels,
    epochs=7,
    batch_size=512,
    validation_split=0.2,
	callbacks=[early_stopping])

test_loss, test_acc = classifier.evaluate(test_data, test_labels)

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')