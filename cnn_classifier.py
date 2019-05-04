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
		self._global_pooling = layers.GlobalAveragePooling1D()
		self._dense_1 = layers.Dense(16, activation='relu')
		self._dense_fc = layers.Dense(1, activation='sigmoid')

	def call(self, x):

		emb_layer = self._embedding(x)
		global_pooling = self._global_pooling(emb_layer)
		dense_1 = self._dense_1(global_pooling)
		dense_fc = self._dense_fc(dense_1)

		return dense_fc

classifier = SimpleClassifier(MAXLEN, EMB_DIM, VOC_SIZE)

classifier.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = classifier.fit(
    train_data,
    train_labels,
    epochs=10,
    batch_size=512,
    validation_split=0.2)

test_loss, test_acc = classifier.evaluate(test_data, test_labels)

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')


seq_input = layers.Input(shape=(MAXLEN,), dtype='int32')
embed_layer = tf.keras.layers.Embedding(VOC_SIZE, EMB_DIM)(seq_input)
# <tf.Tensor 'embedding_7/embedding_lookup/Identity_2:0' shape=(None, 20, 128) dtype=float32>

embedded_layer = layers.Reshape((MAXLEN, EMB_DIM, 1))(embed_layer)
filter_5 = layers.Conv2D(100, (5, EMB_DIM), activation='relu')(embedded_layer) # filters, kernel size
filter_5 = layers.MaxPooling2D((MAXLEN - 5 + 1, 1))(filter_5)

filter_4 = layers.Conv2D(100, (4, EMB_DIM), activation='relu')(embedded_layer) # filters, kernel size
filter_4 = layers.MaxPooling2D((MAXLEN - 4 + 1, 1))(filter_4)

filter_3 = layers.Conv2D(100, (3, EMB_DIM), activation='relu')(embedded_layer) # filters, kernel size
filter_3 = layers.MaxPooling2D((MAXLEN - 3 + 1, 1))(filter_3)

alpha = layers.concatenate([filter_3, filter_4, filter_5])
alpha = layers.Flatten()(alpha)
alpha = layers.Dropout(0.5)(alpha)

# preds = layers.Dense(len(), activation='softmax')(alpha)
preds = layers.Dense(1, activation='sigmoid')(alpha)

model = tf.keras.models.Model(seq_input, preds)
adadelta = tf.keras.optimizers.Adadelta()

# model.compile(loss='categorical_crossentropy',
#               optimizer=adadelta,
#               metrics=['accuracy'])

model.compile(loss='binary_crossentropy',
              optimizer=adadelta,
              metrics=['accuracy'])

history = model.fit(
    train_data,
    train_labels,
    epochs=30,
    batch_size=512,
    validation_split=0.2)


test_loss, test_acc = model.evaluate(test_data)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

# def pad_to_size(vec, size):
#     zeros = [0] * (size - len(vec))
#     vec.extend(zeros)
#     return vec
#
# def sample_predict(sentence, pad):
#     tokenized_sample_pred_text = tokenizer.encode(sample_pred_text)
#
#     if pad:
#         tokenized_sample_pred_text = pad_to_size(tokenized_sample_pred_text, 64)
#
#     predictions = model.predict(tf.expand_dims(tokenized_sample_pred_text, 0))
#
#     return (predictions)
#
# # predict on a sample text without padding.
#
# sample_pred_text = ('The movie was cool. The animation and the graphics '
#                     'were out of this world. I would recommend this movie.')
# predictions = sample_predict(sample_pred_text, pad=False)
# print (predictions)
#
# # predict on a sample text with padding
#
# sample_pred_text = ('The movie was cool. The animation and the graphics '
#                     'were out of this world. I would recommend this movie.')
# predictions = sample_predict(sample_pred_text, pad=True)
# print (predictions)
#
# plot_graphs(history, 'accuracy')
# plot_graphs(history, 'loss')