from gensim.models.word2vec import Word2Vec
import pandas as pd
from gensim.models import word2vec


data = pd.read_csv('data/train.txt', delimiter='\t', header=0)
data = list(data['document'])
model = word2vec.Word2Vec(data, sg=1, size=100, window=3, min_count=5, workers=-1)
print(data)
model.init_sims(replace=True)

model_name = 'naver_sentiment_word2vec'
model.save(model_name)