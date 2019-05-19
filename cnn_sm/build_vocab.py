import pandas as pd
import itertools
import pickle
from pathlib import Path
from konlpy.tag import Okt
from soynlp.tokenizer import LTokenizer
import sys
sys.path.append('..')
from mxnet import gluon
import gluonnlp as nlp
import gensim
from gensim.models import FastText


# load data
proj_dir = Path.cwd()
tr_filepath = proj_dir / 'data' / 'train.txt'
tr = pd.read_csv(tr_filepath, sep='\t').loc[:, ['document', 'label']]

# extracting morph in sentences
def select_tokenizer(model):
    if model == Okt:
        tokenizer = Okt()
        tokenized = tr['document'].apply(tokenizer.morphs).tolist()
    if model == LTokenizer:
        tokenizer = LTokenizer()
        tokenized = tr['document'].apply(tokenizer.morphs).tolist()

    return tokenized

print(select_tokenizer(Okt))
tokenized = select_tokenizer(Okt)

# making the vocab
vocab = gensim.models.Word2Vec.load('ko.bin')
counter = vocab.corpus_count
print(counter)
vocab = FastText(tokenized, size=100, window=5, min_count=5, workers=4, sg=1)

# saving vocab
with open('./data/vocab1.pkl', mode='wb') as io:
    pickle.dump(vocab, io)