import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ops import Flatten, Permute
from gluonnlp import Vocab


class EfficientCharCRNN(nn.Module):
    def __init__(self, args, vocab, word_dropout_ratio: float = .5):
        super(EfficientCharCRNN, self).__init__()
        self._dim = args.word_dim
        self._word_dropout_ratio = word_dropout_ratio
        self._embedding = nn.Embedding(len(vocab), self._dim, vocab.to_indices(vocab.padding_token))
        self._conv = nn.Conv1d(in_channels=self._dim, out_channels=128, kernel_size=5, stride=1, padding=1)
        self._conv1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self._maxpool = nn.MaxPool1d(2, stride=2)
        self._maxpool1 = nn.MaxPool1d(2, stride=2)
        self._dropout = nn.Dropout()
        self._bilstm = nn.LSTM(128, 128, dropout=self._word_dropout_ratio, batch_first=True, bidirectional=True)
        self._fc = nn.Linear(256, args.classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            m = x.bernoulli(self._word_dropout_ratio)
            x = torch.where(m == 1, torch.tensor(0).to(x.device), x)

        embedding = self._embedding(x).permute(0, 2, 1)

        r = self._conv(embedding)
        r = F.relu(r)
        r = self._maxpool(r)
        r = self._conv1(r)
        r = F.relu(r)
        r = self._maxpool1(r)
        r = r.permute(0, 2, 1)

        _, r = self._bilstm(r)
        feature = torch.cat([*r[0]], dim=1)
        r = self._dropout(feature)
        score = self._fc(r)
        return score
