embedding_dim = 50
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

# Training parameters
batch_size = 64
num_epochs = 10


class TextCNN(tf.keras.Model):
    """
    <Parameters>
        - sequence_length: 최대 문장 길이
        - num_classes: 클래스 개수
        - vocab_size: 등장 단어 수
        - embedding_size: 각 단어에 해당되는 임베디드 벡터의 차원
        - filter_sizes: convolutional filter들의 사이즈 (= 각 filter가 몇 개의 단어를 볼 것인가?) (예: "3, 4, 5")
        - num_filters: 각 filter size 별 filter 수

    """
    def __init__(self, max_len, emb_dim, vocab_size):
        
        super(TextCNN, self).__init__()
        self.MAX_LEN = max_len
        self.EMB_DIM = emb_dim
        self.VOC_SIZE = vocab_size
        
        self._embedding = keras.layers.Embedding(VOC_SIZE, self.EMB_DIM, input_length=self.MAX_LEN)
        self._reshape = keras.layers.Reshape((self.MAX_LEN, self.EMB_DIM, 1))

        self._cnn_filter_3 = keras.layers.Conv2D(100, kernel_size=(3, self.EMB_DIM), padding='valid',
                                           kernel_initializer='normal', activation='relu') # filters, kernel size
        self._max_pool_3 = keras.layers.MaxPooling2D((self.MAX_LEN - 3 + 1, 1), strides=(1,1), padding='valid')

        self._cnn_filter_4 = keras.layers.Conv2D(100, (4, self.EMB_DIM), padding='valid',
                                           kernel_initializer='normal', activation='relu') # filters, kernel size
        self._max_pool_4 = keras.layers.MaxPooling2D((self.MAX_LEN - 4 + 1, 1), strides=(1,1), padding='valid')

        self._cnn_filter_5 = keras.layers.Conv2D(100, (5, self.EMB_DIM), padding='valid',
                                           kernel_initializer='normal', activation='relu') # filters, kernel size
        self._max_pool_5 = keras.layers.MaxPooling2D((self.MAX_LEN - 5 + 1, 1), strides=(1,1), padding='valid')
        
        # Fully connected
        self._fc_dense = keras.layers.Flatten()
        self._dropout = keras.layers.Dropout(0.5)
        self._dense_out = keras.layers.Dense(1, activation='sigmoid')


    
    def call(self, x):

        emb_layer = self._embedding(x)
        emb_layer = self._reshape(emb_layer)

        cnn_1 = self._cnn_filter_3(emb_layer)
        max_1 = self._max_pool_3(cnn_1)
        cnn_2 = self._cnn_filter_4(emb_layer)
        max_2 = self._max_pool_4(cnn_2)
        cnn_3 = self._cnn_filter_5(emb_layer)
        max_3 = self._max_pool_5(cnn_3)

        concat = keras.layers.concatenate([max_1, max_2, max_3])
        dense_fc = self._fc_dense(concat)
        drop_out = self._dropout(dense_fc)
        dense_out = self._dense_out(drop_out)

        return dense_out
    
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train,
                                                        value=1,
                                                        padding='post',
                                                        maxlen=MAXLEN)
       

classifier = TextCNN(MAXLEN, EMB_DIM, VOC_SIZE)
classifier.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = classifier.fit(
    X_train,
    y_train,
    epochs=7,
    batch_size=512)

test_input_data = tf.keras.preprocessing.sequence.pad_sequences(test_input_data,
                                                                value=1,
                                                                padding='post',
                                                                maxlen=MAXLEN)

test_loss, test_acc = classifier.evaluate(test_data, test_labels)