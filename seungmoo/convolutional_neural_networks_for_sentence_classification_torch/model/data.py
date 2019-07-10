import pandas as pd
import torch
from torch.utils.data import Dataset


class Corpus(Dataset):
    def __init__(self, filepath, transform_fn):
        self._corpus = pd.read_csv(filepath, sep='\t').loc[:, ['document', 'label']]
        self._transform = transform_fn

    def __len__(self):
        return len(self._corpus)

    def __getitem__(self, idx):
        tokens2indices = torch.tensor(self._transform(self._corpus.iloc[idx]['document']))
        label = torch.tensor(self._corpus.iloc[idx]['label'])
        return tokens2indices, label
