"""Small word-level LSTM LM shared by training and the neural abbreviation codec.

Context is reset every CHUNK words so that the decoder can decode independent
chunks in parallel (and the encoder can run teacher-forced chunks in parallel).
"""
import json, os
from collections import Counter
import numpy as np
import torch
import torch.nn as nn

V_MAX, MAXLEN, CHUNK = 20000, 12, 256
N_TRAIN_WORDS = 15_000_000       # of the first 90M chars; rest of that segment = validation
TRAIN_CHARS = 90_000_000
EMB, HID = 256, 512


def load_words():
    text = open(os.path.join(os.path.dirname(__file__), "data/text8")).read()
    train_all = text[:TRAIN_CHARS].split()
    test = text[TRAIN_CHARS:].split()[1:]
    return train_all[:N_TRAIN_WORDS], train_all[N_TRAIN_WORDS:], test


def build_vocab(train):
    cnt = Counter(train)
    top = [w for w, _ in cnt.most_common() if len(w) <= MAXLEN][:V_MAX]
    return ["<unk>"] + top


def to_ids(words, w2i):
    return np.fromiter((w2i.get(w, 0) for w in words), dtype=np.int64, count=len(words))


class LM(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.V = V                       # output classes (0 = unk); input has extra BOS = V
        self.emb = nn.Embedding(V + 1, EMB)
        self.rnn = nn.LSTM(EMB, HID, batch_first=True)
        self.proj = nn.Linear(HID, EMB)
        self.bias = nn.Parameter(torch.zeros(V))
        self.drop = nn.Dropout(0.2)

    def logits_from_h(self, h):
        return self.proj(self.drop(h)) @ self.emb.weight[: self.V].T + self.bias

    def forward(self, inp, state=None):
        h, state = self.rnn(self.drop(self.emb(inp)), state)
        return self.logits_from_h(h), state


def make_chunks(ids, V):
    """ids -> (inputs, targets) arrays of shape (n_chunks, CHUNK); drops the tail."""
    n = len(ids) // CHUNK
    tgt = ids[: n * CHUNK].reshape(n, CHUNK)
    inp = np.concatenate([np.full((n, 1), V, dtype=np.int64), tgt[:, :-1]], axis=1)
    return inp, tgt
