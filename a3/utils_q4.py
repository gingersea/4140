"""Data and one-layer models for the noisy sine experiment."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset


def data_generator(noise_parameter, seed=42):
    x = np.arange(1000) / 10
    noise = np.random.default_rng(seed).normal(0, noise_parameter, len(x))
    return x, (np.sin(x) + noise).astype(np.float32)


class MyDataset(TensorDataset):
    def __init__(self, series, seq_len=10):
        windows = series.unfold(0, seq_len + 1, 1)
        super().__init__(windows[:, :-1].unsqueeze(-1).contiguous(), windows[:, -1])


class RNN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(1, hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, hidden = self.rnn(x)
        return self.readout(hidden[-1]).squeeze(-1)


class GRU(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gru = nn.GRU(1, hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.readout(hidden[-1]).squeeze(-1)


class Attention(nn.Module):
    def __init__(self, hidden_size, seq_len=10):
        super().__init__()
        self.embedding = nn.Linear(1, hidden_size)
        self.position = nn.Parameter(torch.randn(seq_len, hidden_size) * 0.1)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=1, batch_first=True)
        self.readout = nn.Linear(hidden_size, 1)

    def forward(self, x):
        tokens = self.embedding(x) + self.position
        context, _ = self.attn(tokens[:, -1:], tokens, tokens, need_weights=False)
        return self.readout(context[:, 0]).squeeze(-1)


class LogisticRegression(nn.Module):
    def __init__(self, seq_len=10):
        super().__init__()
        self.linear = nn.Linear(seq_len, 1)

    def forward(self, x):
        score = self.linear(x.squeeze(-1)).squeeze(-1)
        return 4 * torch.sigmoid(score) - 2
