import torch
import torch.nn as nn
import torch.nn.functional as F
import gluonnlp as nlp
from gluonnlp import Vocab
from typing import Tuple


class MultiChannelEmbedding(nn.Module):



class ConvAndPooling(nn.Module):
    def __init__(self, embedding_dim, dropout_ratio, filters=300):
        super(ConvAndPooling, self).__init__()
        self._tri_gram_ops = nn.Conv1d(in_channels=1, out_channels=filters // 3,
                                       kernel_size=(3, embedding_dim), bias=False)
        self._tetra_gram_ops = nn.Conv1d(in_channels=1, out_channels=filters // 3,
                                         kernel_size=(4, embedding_dim), bias=False)
        self._penta_gram_ops = nn.Conv1d(in_channels=1, out_channels=filters // 3,
                                         kernel_size=(5, embedding_dim), bias=False)
        self.dropout = nn.Dropout(dropout_ratio)
        self.fc1 = nn.Linear(3 * filters, )

    def forward(self, x):
        x = self.pool(F.relu(self._tri_gram_ops(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




