import numpy as np
import pytest

from cllm.compressors import build_compressor
from cllm.compressors.lm_codec import RULES, subseq_candidates
from cllm.config import deep_merge, load_config, stage_hash
from cllm.flops import gpt_train_flops_per_token, lstm_lm_fwd_flops_per_token
from cllm.tokenizer import KINDS, encode_words, ngram_proxy_bits, roundtrip_rate, to_lines, train_tokenizer


def test_subseq_candidates_shortest_first_and_keep_first_letter():
    c = list(subseq_candidates("unique"))
    assert c[0] == "u"
    assert [len(x) for x in c] == sorted(len(x) for x in c)
    assert "uiqu" in c and c[-1] == "unique"
    assert all(x[0] == "u" for x in c)


def test_rules():
    assert RULES["vowel_drop"]("algorithm") == "algrthm"
    assert RULES["vowel_drop"]("area") == "ar"
    assert RULES["trunc"]("algorithm", 3) == "alg"


def test_config_extends_and_overrides(tmp_path):
    (tmp_path / "a.yaml").write_text("x: {p: 1, q: 2}\ny: 3\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\nx: {q: 5}\n")
    cfg = load_config(tmp_path / "b.yaml", ["y=7", "x.z=true"])
    assert cfg["x"] == {"p": 1, "q": 5, "z": True} and cfg["y"] == 7 and cfg["name"] == "b"
    assert deep_merge({"a": {"b": 1}}, {"a": {"c": 2}}) == {"a": {"b": 1, "c": 2}}
    assert stage_hash({"a": 1}) == stage_hash({"a": 1}) != stage_hash({"a": 2})


def test_flops_scale_sensibly():
    small = gpt_train_flops_per_token(2, 128, 128, 1000)
    big = gpt_train_flops_per_token(4, 256, 128, 1000)
    assert 0 < small < big
    assert lstm_lm_fwd_flops_per_token(1000, 64, 128) > 0


def test_identity_roundtrip_and_registry():
    c = build_compressor({"kind": "identity"})
    assert c.decompress(c.compress(["a", "b"])) == ["a", "b"]
    with pytest.raises(ValueError):
        build_compressor({"kind": "nope"})


@pytest.mark.parametrize("kind", KINDS)
def test_tokenizer_roundtrip(kind):
    rng = np.random.default_rng(0)
    pool = ["algrthm", "uiqu", "#hello", "th", "of", "and", "wrld", "x", "qz"]
    words = [pool[i] for i in rng.integers(0, len(pool), 5000)]
    tok = train_tokenizer(to_lines(words), kind, 100, min_frequency=1)
    assert roundtrip_rate(tok, words, n_lines=50) == 1.0
    ids = encode_words(tok, words)
    assert ids.dtype == np.uint16 and len(ids) >= len(words) / 2


def test_ngram_proxy_prefers_predictable_streams():
    rng = np.random.default_rng(1)
    periodic = np.tile(np.arange(10), 2000)
    noisy = rng.integers(0, 10, 20000)
    p = ngram_proxy_bits(periodic[:15000], periodic[15000:], 10, 20000)
    n = ngram_proxy_bits(noisy[:15000], noisy[15000:], 10, 20000)
    assert p["bigram_bits_per_char"] < n["bigram_bits_per_char"]


def test_unique_codec_is_lossless_and_only_drops_unambiguous_letters():
    from cllm.compressors.unique import UniqueCodec
    train = ["unique", "uniform", "the", "this", "that", "there", "algorithm", "cat", "cart"] * 50
    c = UniqueCodec(vocab_size=100).fit(train)
    toks = c.compress(["unique", "the", "algorithm", "zzz"])
    assert toks[3] == "#zzz"                                  # OOV is escaped
    assert toks[1] == "the"                                   # `t`/`th`/`te` also fit this/that/there
    assert len(toks[0]) < len("unique") and len(toks[2]) < len("algorithm")
    words = ["unique", "uniform", "the", "this", "that", "there", "algorithm", "cat", "cart", "qq"]
    assert c.decompress(c.compress(words)) == words
    # every abbreviation matches exactly one vocabulary word as a subsequence
    from cllm.compressors.lm_codec import subseq_candidates
    for w, t in c.enc.items():
        if t != w:
            assert sum(t in set(subseq_candidates(v)) for v in c.enc) == 1
