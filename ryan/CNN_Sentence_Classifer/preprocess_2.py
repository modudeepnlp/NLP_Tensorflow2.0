import pandas as pd

from pathlib import Path
from konlpy.tag import Mecab
import json
import re
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np


# loading dataset
proj_dir = Path.cwd()
params = json.load((proj_dir / 'params' / 'config.json').open())
train_path = params['filepath'].get('tr')
test_path = params['filepath'].get('test')

train_data = pd.read_csv(train_path, header=0, delimiter='\t', quoting=3 )
train_data = train_data.dropna()

def preprocessing(review, mecab, remove_stopwords=False, stop_words=[]):
	# 함수의 인자는 다음과 같다.
	# review : 전처리할 텍스트
	# okt : okt 객체를 반복적으로 생성하지 않고 미리 생성후 인자로 받는다.
	# remove_stopword : 불용어를 제거할지 선택 기본값은 False
	# stop_word : 불용어 사전은 사용자가 직접 입력해야함 기본값은 비어있는 리스트

	# 1. 한글 및 공백을 제외한 문자 모두 제거.
	review_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", "", review)

	# 2. okt 객체를 활용해서 형태소 단위로 나눈다.
	word_review = mecab.morphs(review_text)

	if remove_stopwords:
		# 불용어 제거(선택적)
		word_review = [token for token in word_review if not token in stop_words]

	return word_review

stop_words = [ '은', '는', '이', '가', '하', '아', '것', '들','의', '있', '되', '수', '보', '주', '등', '한']
mecab = Mecab() #mecab 불러오기
clean_train_review = []

for review in train_data['document']:
    # 비어있는 데이터에서 멈추지 않도록 string인 경우만 진행
    if type(review) == str:
        clean_train_review.append(preprocessing(review, mecab, remove_stopwords = True, stop_words=stop_words))
    else:
        clean_train_review.append([])  #string이 아니면 비어있는 값 추가

test_data = pd.read_csv(test_path, header=0, delimiter='\t', quoting=3 )
test_data = test_data.dropna()

clean_test_review = []

for review in test_data['document']:
    # 비어있는 데이터에서 멈추지 않도록 string인 경우만 진행
    if type(review) == str:
        clean_test_review.append(preprocessing(review, mecab, remove_stopwords = True, stop_words=stop_words))
    else:
        clean_test_review.append([])  #string이 아니면 비어있는 값 추가

tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_train_review)
train_sequences = tokenizer.texts_to_sequences(clean_train_review)
test_sequences = tokenizer.texts_to_sequences(clean_test_review)

word_vocab = tokenizer.word_index # 단어 사전 형태

MAX_SEQUENCE_LENGTH = 15 # 문장 최대 길이

train_inputs = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post') # 학습 데이터를 벡터화
train_labels = np.array(train_data['label']) # 학습 데이터의 라벨

test_inputs = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post') # 테스트 데이터를 벡터화
test_labels = np.array(test_data['label']) # 테스트 데이터의 라벨

DATA_IN_PATH = './data/'
TRAIN_INPUT_DATA = 'nsmc_train_input.npy'
TRAIN_LABEL_DATA = 'nsmc_train_label.npy'
TEST_INPUT_DATA = 'nsmc_test_input.npy'
TEST_LABEL_DATA = 'nsmc_test_label.npy'
DATA_CONFIGS = 'data_configs.json'

data_configs = {}

data_configs['vocab'] = word_vocab
data_configs['vocab_size'] = len(word_vocab) # vocab size 추가

import os
# 저장하는 디렉토리가 존재하지 않으면 생성
if not os.path.exists(DATA_IN_PATH):
    os.makedirs(DATA_IN_PATH)

# 전처리 된 학습 데이터를 넘파이 형태로 저장
np.save(open(DATA_IN_PATH + TRAIN_INPUT_DATA, 'wb'), train_inputs)
np.save(open(DATA_IN_PATH + TRAIN_LABEL_DATA, 'wb'), train_labels)
# 전처리 된 테스트 데이터를 넘파이 형태로 저장
np.save(open(DATA_IN_PATH + TEST_INPUT_DATA, 'wb'), test_inputs)
np.save(open(DATA_IN_PATH + TEST_LABEL_DATA, 'wb'), test_labels)

# 데이터 사전을 json 형태로 저장
json.dump(data_configs, open(DATA_IN_PATH + DATA_CONFIGS, 'w'), ensure_ascii=False)