"""Compression-level metrics: reconstruction accuracy, size reduction, speed."""
import bz2
import lzma
import time
import zlib

import numpy as np


def classical_sizes(text):
    b = text.encode()
    return {"raw": len(b), "zlib9": len(zlib.compress(b, 9)), "bz2": len(bz2.compress(b, 9)),
            "xz9": len(lzma.compress(b, preset=9))}


def compression_report(comp, words, workers=0, progress=False):
    """Compress -> decompress `words`; return metrics and the compressed token list."""
    t0 = time.time()
    toks = comp.compress(words, workers=workers, progress=progress)
    t_comp = time.time() - t0
    t0 = time.time()
    back = comp.decompress(toks, progress=progress)
    t_dec = time.time() - t0
    n = len(words)
    assert len(toks) == n, "compressors must emit exactly one token per word"
    wrong = [i for i, (a, b) in enumerate(zip(back, words)) if a != b]
    wrong_chars = sum(len(words[i]) for i in wrong)
    orig, comp_text = " ".join(words), " ".join(toks)
    escapes = sum(t.startswith("#") for t in toks)
    s_orig, s_comp = classical_sizes(orig), classical_sizes(comp_text)
    # mean token length / word length, by original word length
    by_len = {}
    for w, t in zip(words, toks):
        by_len.setdefault(min(len(w), 12), []).append(len(t) / len(w))
    return {
        "n_words": n,
        "orig_chars": len(orig),
        "comp_chars": len(comp_text),
        "char_ratio": len(comp_text) / len(orig),
        "chars_per_word_orig": len(orig) / n,
        "chars_per_word_comp": len(comp_text) / n,
        "escape_rate": escapes / n,
        "lossless": len(back) == n and not wrong,
        "word_accuracy": 1 - len(wrong) / n,
        "char_accuracy": 1 - wrong_chars / max(len(orig), 1),
        "sizes_orig": s_orig,
        "sizes_comp": s_comp,
        "size_ratio_comp_over_orig": {k: s_comp[k] / s_orig[k] for k in s_orig},
        "compress_seconds": t_comp,
        "decompress_seconds": t_dec,
        "compress_words_per_s": n / max(t_comp, 1e-9),
        "decompress_words_per_s": n / max(t_dec, 1e-9),
        "token_len_ratio_by_word_len": {str(k): float(np.mean(v)) for k, v in sorted(by_len.items())},
    }, toks
