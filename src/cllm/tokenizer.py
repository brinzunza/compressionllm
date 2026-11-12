"""Tokenizers trained on (compressed) text, plus the metrics used to iterate on them.

All kinds split on spaces, so a compressed-word boundary is never ambiguous and the token stream
decodes back to exactly the space-joined word stream.

  bpe       byte-pair encoding over characters (Metaspace pre-tokenizer)
  unigram   unigram LM tokenizer (Metaspace)
  wordlevel one token per distinct compressed word (rare words -> <unk>, which is lossy)
  char      one token per character (vocab = the ~30 characters seen in training). Here the sequence
            length equals the compressed character count, so letter dropping shortens sequences
            one for one, unlike with BPE where it mostly just fragments the tokens.
"""
import time

import numpy as np
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

SPECIALS = ["<unk>", "<eos>"]
KINDS = ("bpe", "unigram", "wordlevel", "char")


def to_lines(words, line_words=64):
    return [" ".join(words[i:i + line_words]) for i in range(0, len(words), line_words)]


def _char_tokenizer(lines):
    chars = sorted({c for line in lines for c in line} | {" "})
    vocab = {t: i for i, t in enumerate(SPECIALS + chars)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.normalizer = normalizers.Prepend(" ")        # lines are encoded separately; the leading space is the
    tok.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")   # separator between them
    tok.decoder = decoders.Sequence([decoders.Fuse(), decoders.Strip(content=" ", left=1, right=0)])
    return tok


def train_tokenizer(lines, kind="bpe", vocab_size=8000, min_frequency=2):
    if kind == "char":
        return _char_tokenizer(lines)
    if kind == "bpe":
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS,
                                      min_frequency=min_frequency, show_progress=False)
    elif kind == "unigram":
        tok = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(vocab_size=vocab_size, special_tokens=SPECIALS,
                                          unk_token="<unk>", max_piece_length=16, show_progress=False)
    elif kind == "wordlevel":
        tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
        trainer = trainers.WordLevelTrainer(vocab_size=vocab_size, special_tokens=SPECIALS,
                                            min_frequency=min_frequency, show_progress=False)
    else:
        raise ValueError(f"unknown tokenizer kind {kind}; choose from {KINDS}")
    if kind == "wordlevel":
        tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    else:
        tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
        tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    tok.train_from_iterator(lines, trainer)
    return tok


def encode_words(tok, words, line_words=64, batch=2048):
    """Encode a word stream to one flat id array (lines are word-aligned)."""
    lines = to_lines(words, line_words)
    parts = []
    for i in range(0, len(lines), batch):
        parts.extend(np.asarray(e.ids, dtype=np.int64) for e in tok.encode_batch(lines[i:i + batch]))
    ids = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    return ids.astype(np.uint16 if tok.get_vocab_size() < 65536 else np.uint32)


def decode_ids(tok, ids):
    return tok.decode([int(i) for i in ids], skip_special_tokens=False)


def roundtrip_rate(tok, words, n_lines=2000, line_words=64):
    """Fraction of lines that survive encode -> decode unchanged."""
    lines = to_lines(words, line_words)[:n_lines]
    ok = 0
    for line, e in zip(lines, tok.encode_batch(lines)):
        ok += tok.decode(e.ids, skip_special_tokens=False) == line
    return ok / max(len(lines), 1)


def ngram_proxy_bits(train_ids, test_ids, V, orig_chars):
    """Cheap tokenizer-quality proxies: held-out cross-entropy (bits per ORIGINAL character) of a
    unigram model and of an interpolated bigram model estimated on the training ids.
    Lower is better; unlike tokens/char it accounts for how predictable the tokens are."""
    tr, te = train_ids.astype(np.int64), test_ids.astype(np.int64)
    uni = np.bincount(tr, minlength=V) + 0.5
    pu = uni / uni.sum()
    p_uni = pu[te]
    keys = tr[:-1] * V + tr[1:]
    uk, uc = np.unique(keys, return_counts=True)
    ctx = np.bincount(tr[:-1], minlength=V)
    tk = te[:-1] * V + te[1:]
    pos = np.minimum(np.searchsorted(uk, tk), len(uk) - 1)
    hit = uk[pos] == tk
    c = np.where(hit, uc[pos], 0)
    prev_ctx = ctx[te[:-1]]
    lam = np.where(prev_ctx > 0, 0.8, 0.0)
    p_bi = lam * (c / np.maximum(prev_ctx, 1)) + (1 - lam) * pu[te[1:]]
    return {
        "unigram_bits_per_char": float(-np.log2(p_uni).sum() / orig_chars),
        "bigram_bits_per_char": float((-np.log2(p_bi).sum() - np.log2(p_uni[0])) / orig_chars),
    }


def tokenizer_report(tok, train_ids, test_ids, test_words, orig_chars, orig_words):
    """Metrics for one tokenizer on held-out text. `test_words` is the (compressed) word stream."""
    V = tok.get_vocab_size()
    comp_chars = sum(len(w) for w in test_words) + max(len(test_words) - 1, 0)
    unk = int((test_ids == 0).sum())
    used = int(len(np.unique(test_ids)))
    vocab = tok.get_vocab()
    lens = np.array([len(t.replace("▁", "")) for t in vocab], dtype=np.float64)
    rep = {
        "vocab_size": V,
        "tokens": int(len(test_ids)),
        "tokens_per_orig_char": len(test_ids) / orig_chars,
        "tokens_per_orig_word": len(test_ids) / orig_words,
        "compressed_chars_per_token": comp_chars / max(len(test_ids), 1),
        "unk_rate": unk / max(len(test_ids), 1),
        "vocab_used_frac": used / V,
        "mean_vocab_piece_len": float(lens.mean()),
        "roundtrip_lines_ok": roundtrip_rate(tok, test_words),
    }
    rep.update(ngram_proxy_bits(train_ids, test_ids, V, orig_chars))
    return rep


def sweep(train_lines, train_words, test_words, orig_chars, orig_words, kinds, vocab_sizes, source):
    """Train one tokenizer per (kind, vocab_size) and report; used to iterate on tokenizer design."""
    rows = []
    for kind in kinds:
        for vs in vocab_sizes:
            t0 = time.time()
            tok = train_tokenizer(train_lines, kind, vs)
            secs = time.time() - t0
            tr_ids = encode_words(tok, train_words)
            te_ids = encode_words(tok, test_words)
            rep = tokenizer_report(tok, tr_ids, te_ids, test_words, orig_chars, orig_words)
            rows.append({"source": source, "kind": kind, "vocab_requested": vs, "train_seconds": secs, **rep})
    return rows
