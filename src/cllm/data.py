"""text8 download + deterministic word-level splits."""
import hashlib
import urllib.request
import zipfile
from pathlib import Path

TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
TEXT8_SHA1 = "0dc3edebc970dcc96137e7deda4d9995af9d93de"


def _sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure_text8(data_dir="data"):
    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    f = d / "text8"
    if not f.exists():
        z = d / "text8.zip"
        urllib.request.urlretrieve(TEXT8_URL, z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(d)
        z.unlink()
    got = _sha1(f)
    if got != TEXT8_SHA1:
        raise RuntimeError(f"text8 checksum mismatch: {got}")
    return f


def load_words(data_cfg):
    """train = first n_train words, val = the next n_val, test = the last n_test (all disjoint)."""
    f = ensure_text8(data_cfg.get("dir", "data"))
    words = f.read_text().split()
    n_tr, n_va, n_te = data_cfg["n_train_words"], data_cfg["n_val_words"], data_cfg["n_test_words"]
    if n_tr + n_va + n_te > len(words):
        raise ValueError(f"requested {n_tr + n_va + n_te} words but text8 has {len(words)}")
    return {
        "train": words[:n_tr],
        "val": words[n_tr:n_tr + n_va],
        "test": words[len(words) - n_te:],
    }


def n_chars(words):
    """Characters of the original text for a word list (words joined by single spaces)."""
    return sum(len(w) for w in words) + max(len(words) - 1, 0)
