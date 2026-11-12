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
    assert ids.dtype == np.uint16 and (kind == "phrase" or len(ids) >= len(words) / 2)


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


def _phrase_corpus(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    common = ["the", "of", "and", "in", "city", "river", "king", "war"]
    rare = ["zoology", "quasar", "xylophone", "mnemonic"]
    out = []
    while len(out) < n:
        out += ["one", "of", "the", "best"] if rng.random() < 0.4 else [common[rng.integers(8)]]
        if rng.random() < 0.05:
            out.append(rare[rng.integers(4)])
    return out[:n]


def test_phrase_tokenizer_roundtrip_phrases_fallback_and_save(tmp_path):
    from cllm.phrase_tokenizer import K_PHRASE, PhraseTokenizer
    words = _phrase_corpus()
    lines = to_lines(words)
    tok = PhraseTokenizer.fit(lines, words=12, phrases=5, max_n=3, fallback=30, min_count=5, verbose=False)
    assert ("one", "of", "the") in tok.phrase_id or ("of", "the") in tok.phrase_id   # the planted phrase is found
    for kind_words in (words[:200], ["zoology", "unseenword", "of", "the", "quasar"]):    # incl. rare + unseen words
        line = " ".join(kind_words)
        ids = tok.encode(line).ids
        assert tok.decode(ids) == line
    ids = encode_words(tok, words)
    assert len(ids) < 0.9 * len(words)                                   # multi-word tokens cut the token count
    assert (tok.kind[ids] == K_PHRASE).any()
    assert roundtrip_rate(tok, words, n_lines=40) == 1.0
    tok.save(tmp_path / "t.json")
    from cllm.tokenizer import load_tokenizer
    t2 = load_tokenizer(tmp_path / "t.json")
    assert t2.get_vocab_size() == tok.get_vocab_size() and list(encode_words(t2, words[:300])) == list(encode_words(tok, words[:300]))


def test_phrase_tokenizer_with_zero_phrases_is_word_level_plus_fallback():
    from cllm.phrase_tokenizer import PhraseTokenizer
    words = _phrase_corpus()
    tok = PhraseTokenizer.fit(to_lines(words), words=8, phrases=0, max_n=3, fallback=30, verbose=False)
    assert tok.phrases == [] and tok.decode(tok.encode("the zoology of king").ids) == "the zoology of king"
    ids = encode_words(tok, ["the", "of", "king"])
    assert len(ids) == 3                                                  # one token per in-vocabulary word
