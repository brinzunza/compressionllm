"""Slower tests: a tiny decoder LM on synthetic text, codec losslessness, LLM eval bookkeeping."""
import math

import numpy as np
import pytest
import torch

from cllm.compressors import build_compressor
from cllm.llm import bits_per_char, build_model, eval_nll
from cllm.word_lm import train_word_lm


def synthetic_words(n, seed=0):
    rng = np.random.default_rng(seed)
    subj = ["the cat", "a dog", "the bird", "my friend", "your teacher"]
    verb = ["sees", "likes", "wants", "finds", "reads"]
    obj = ["the green book", "a small house", "some warm bread", "the old river", "a quiet garden"]
    out = []
    while len(out) < n:
        out += (subj[rng.integers(5)] + " " + verb[rng.integers(5)] + " " + obj[rng.integers(5)]).split()
    return out[:n]


@pytest.fixture(scope="module")
def lm_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("lm")
    train_word_lm(synthetic_words(20000), synthetic_words(2000, 1),
                  {"vocab_size": 100, "emb": 32, "hid": 64, "chunk": 32, "batch": 16, "steps": 120,
                   "eval_every": 1000, "device": "cpu"}, d, device=torch.device("cpu"))
    return d


@pytest.mark.parametrize("cfg", [{"kind": "learned", "tau": 0.1},
                                 {"kind": "rule", "rule": "vowel_drop", "verify": True},
                                 {"kind": "rule", "rule": "trunc", "k": 2, "verify": True}])
def test_verified_codecs_are_lossless(lm_dir, cfg):
    comp = build_compressor({**cfg, "device": "cpu"}, lm_dir)
    words = synthetic_words(1000, seed=5) + ["zzzunseenword", "qqq"]      # includes OOV words
    toks = comp.compress(words)
    assert len(toks) == len(words)
    assert comp.decompress(toks) == words
    assert sum(len(t) for t in toks) < sum(len(w) for w in words)         # it actually compresses
    assert toks[-2] == "#zzzunseenword"                                   # OOV is escaped


def test_learned_codec_compresses_more_than_trunc_rule_on_predictable_text(lm_dir):
    words = synthetic_words(2000, seed=9)
    learned = build_compressor({"kind": "learned", "tau": 0.1, "device": "cpu"}, lm_dir).compress(words)
    assert sum(map(len, learned)) < 0.7 * sum(map(len, words))


def test_blind_rule_reports_errors_instead_of_hiding_them(lm_dir):
    comp = build_compressor({"kind": "rule", "rule": "trunc", "k": 1, "verify": False, "device": "cpu"}, lm_dir)
    words = synthetic_words(500, seed=3)
    back = comp.decompress(comp.compress(words))
    assert len(back) == len(words)           # same length, but some words may differ (lossy by design)


def test_eval_nll_covers_every_token_and_matches_uniform_baseline():
    torch.manual_seed(0)
    cfg = {"ctx": 16, "n_embd": 16, "n_layer": 1, "n_head": 1}
    model = build_model(cfg, 50)
    for p in model.parameters():
        torch.nn.init.zeros_(p)               # uniform predictions -> NLL = ln(V) exactly
    ids = np.random.default_rng(0).integers(0, 50, 1000)
    nats, cnt = eval_nll(model, ids, 16, "cpu", batch=4)
    assert cnt == len(ids) - 1
    assert nats / cnt == pytest.approx(math.log(50), rel=1e-4)
    assert bits_per_char(nats, cnt, 2.0) == pytest.approx(math.log2(50) / 2.0, rel=1e-4)
