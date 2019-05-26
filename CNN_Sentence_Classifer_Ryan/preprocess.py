import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
import json

# loading dataset
proj_dir = Path.cwd()
params = json.load((proj_dir / 'params' / 'config.json').open())
train_path = params['filepath'].get('tr')

dataset = pd.read_csv(train_path, sep='\t').loc[:, ['document', 'label']]
dataset = dataset.loc[dataset['document'].isna().apply(lambda elm: not elm), :]
tr, val = train_test_split(dataset, test_size=0.2, random_state=777)

tr.to_csv(proj_dir / 'data' / 'naver_movie_corpus_kr' / 'train.txt', sep='\t', index=False, header=False)
val.to_csv(proj_dir / 'data' / 'naver_movie_corpus_kr'/ 'val.txt', sep='\t', index=False, header=False)

tst_filepath = proj_dir / 'data' / 'naver_movie_corpus_kr' / 'ratings_test.txt'
tst = pd.read_csv(tst_filepath, sep='\t').loc[:, ['document', 'label']]
tst = tst.loc[tst['document'].isna().apply(lambda elm: not elm), :]
tst.to_csv(proj_dir / 'data' / 'naver_movie_corpus_kr' / 'test.txt', sep='\t', index=False, header=False)