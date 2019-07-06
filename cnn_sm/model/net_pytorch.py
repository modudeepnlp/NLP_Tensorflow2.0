import torch
import torch.nn as nn
import torch.nn.functional as F


class SenCNN(nn.Module):

    def __init__(self, embedding_dim, num_words, num_filters,
                 num_classes, vocab):
        super(SenCNN, self).__init__()
        self._embedding = nn.Embedding(num_words, embedding_dim)

    def forward(self, x):
        pass

