"""Letter-dropping codecs decoded by the word-level LM.

Encoding: every word is replaced by an abbreviation. The decoder looks up all vocabulary words that
map to that abbreviation and picks the one the LM scores highest given the previous decoded words.

Schemes
  subseq  ("learned"): the abbreviation is any subsequence of the word's letters (first letter kept).
          The encoder simulates the decoder and keeps the shortest abbreviation that decodes back to
          the true word with a logit margin >= tau, so drop-ability depends on context. Lossless.
  rule    ("fixed"): a context-free rule (drop non-initial vowels, or truncate to k letters).
          verify=False applies the rule blindly (lossy where the decoder guesses wrong);
          verify=True escapes any word the decoder would get wrong (lossless).

Words that cannot be represented (out of vocabulary, too long, or failing verification) are
escaped as '#word'. The context for the LM is reset every `chunk` words.
"""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import torch

from ..flops import lstm_lm_fwd_flops_per_token
from ..utils import exact_fp32, get_device, log
from ..word_lm import load_word_lm
from .base import Compressor

VOWELS = frozenset("aeiou")


def subseq_candidates(word):
    """Subsequences keeping the first letter, shortest first."""
    n = len(word)
    for k in range(n):
        for comb in combinations(range(1, n), k):
            yield word[0] + "".join(word[i] for i in comb)


def vowel_drop(word, k=None):
    return word[0] + "".join(c for c in word[1:] if c not in VOWELS)


def truncate(word, k=3):
    return word[:k]


RULES = {"vowel_drop": vowel_drop, "trunc": truncate}

_WORKER = None


def _worker_encode(chunks):
    torch.set_num_threads(1)
    return _WORKER._encode_chunks(chunks)


class LMCodec(Compressor):
    needs_lm = True

    def __init__(self, scheme="subseq", rule="vowel_drop", k=3, verify=None, tau=0.1,
                 batch_chunks=8, decode_batch=256, device="auto"):
        if scheme not in ("subseq", "rule"):
            raise ValueError(scheme)
        if scheme == "rule" and rule not in RULES:
            raise ValueError(f"unknown rule {rule}; choose from {list(RULES)}")
        self.scheme, self.rule, self.k = scheme, rule, k
        self.verify = (scheme == "subseq") if verify is None else bool(verify)
        self.tau, self.batch_chunks, self.decode_batch, self.device_pref = tau, batch_chunks, decode_batch, device
        self._ctor = dict(scheme=scheme, rule=rule, k=k, verify=self.verify, tau=tau,
                          batch_chunks=batch_chunks, decode_batch=decode_batch, device=device)
        self.lm_dir = None
        self.vocab = self.w2i = self.meta = self._inv = None
        self.model, self._dev = None, None

    # ---------------- lazy loading ----------------
    def _ensure_vocab(self):
        if self.vocab is None:
            d = Path(self.lm_dir)
            self.meta = json.loads((d / "meta.json").read_text())
            self.vocab = json.loads((d / "vocab.json").read_text())
            self.w2i = {w: i for i, w in enumerate(self.vocab)}
            self.V, self.chunk = len(self.vocab), self.meta["chunk"]

    def _candidates(self, word):
        if self.scheme == "subseq":
            return subseq_candidates(word)
        return [RULES[self.rule](word, self.k)]

    def _ensure_index(self):
        self._ensure_vocab()
        if self._inv is not None:
            return
        inv = {}
        for wid in range(1, self.V):
            w = self.vocab[wid]
            cands = subseq_candidates(w) if self.scheme == "subseq" else [RULES[self.rule](w, self.k)]
            for a in cands:
                inv.setdefault(a, []).append(wid)
        self._inv = {a: np.array(l, dtype=np.int32) for a, l in inv.items()}

    def _ensure_model(self, device=None):
        self._ensure_index()
        dev = torch.device(device) if device is not None else get_device(self.device_pref)
        if self.model is None or self._dev != dev:
            self.model, _, _ = load_word_lm(self.lm_dir, dev)
            self._dev = dev

    # ---------------- encode ----------------
    def _encode_word(self, word, wid, row):
        if wid == 0:
            return "#" + word
        rw = row[wid]
        for a in self._candidates(word):
            ids = self._inv.get(a)
            if ids is None:
                continue
            if not self.verify:
                return a
            if len(ids) > 1:
                v = row[ids].copy()
                v[ids == wid] = -np.inf
                if v.max() >= rw - self.tau:
                    continue
            return a
        return "#" + word

    @torch.no_grad()
    def _encode_chunks(self, chunks):
        """chunks: list of equal-length word lists -> list of token lists."""
        B, n = len(chunks), len(chunks[0])
        ids = np.array([[self.w2i.get(w, 0) for w in c] for c in chunks], dtype=np.int64)
        inp = np.concatenate([np.full((B, 1), self.V, dtype=np.int64), ids[:, :-1]], axis=1)
        with exact_fp32():
            lg, _ = self.model(torch.from_numpy(inp).to(self._dev))
        lg = lg.float().cpu().numpy()
        out = []
        for b in range(B):
            out.append([self._encode_word(chunks[b][j], int(ids[b, j]), lg[b, j]) for j in range(n)])
        return out

    def compress(self, words, workers=0, progress=False):
        self._ensure_model("cpu" if workers else None)
        ch = self.chunk
        full = len(words) // ch
        chunks = [words[i * ch:(i + 1) * ch] for i in range(full)]
        groups = [chunks[i:i + self.batch_chunks] for i in range(0, len(chunks), self.batch_chunks)]
        tail = words[full * ch:]
        results = []
        if workers and groups:
            global _WORKER
            _WORKER = self
            with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
                for gi, res in enumerate(ex.map(_worker_encode, groups)):
                    results.extend(res)
                    if progress and gi % max(1, len(groups) // 10) == 0:
                        log(f"[compress] {gi + 1}/{len(groups)} groups")
            _WORKER = None
        else:
            for gi, g in enumerate(groups):
                results.extend(self._encode_chunks(g))
                if progress and gi % max(1, len(groups) // 10) == 0:
                    log(f"[compress] {gi + 1}/{len(groups)} groups")
        if tail:
            results.extend(self._encode_chunks([tail]))
        return [t for c in results for t in c]

    # ---------------- decode ----------------
    @torch.no_grad()
    def _decode_chunks(self, chunks):
        """chunks: list of equal-length token lists, decoded in lock step -> list of word lists."""
        B, n = len(chunks), len(chunks[0])
        prev = np.full(B, self.V, dtype=np.int64)
        state = None
        out = [[] for _ in range(B)]
        with exact_fp32():
            for t in range(n):
                x = torch.from_numpy(prev).to(self._dev)
                h, state = self.model.rnn(self.model.emb(x)[:, None, :], state)
                lg = self.model.logits_from_h(h[:, 0]).float().cpu().numpy()
                for b in range(B):
                    tk = chunks[b][t]
                    if tk.startswith("#"):
                        w = tk[1:]
                        wid = self.w2i.get(w, 0)
                    else:
                        ids = self._inv.get(tk)
                        if ids is None:          # not a valid abbreviation (e.g. LLM output): keep literal
                            w, wid = tk, 0
                        else:
                            wid = int(ids[np.argmax(lg[b][ids])])
                            w = self.vocab[wid]
                    out[b].append(w)
                    prev[b] = wid
        return out

    def decompress(self, tokens, progress=False):
        self._ensure_model()
        ch = self.chunk
        chunks = [tokens[i:i + ch] for i in range(0, len(tokens), ch)]
        full = [c for c in chunks if len(c) == ch]
        tail = [c for c in chunks if len(c) < ch]
        words = []
        for i in range(0, len(full), self.decode_batch):
            for c in self._decode_chunks(full[i:i + self.decode_batch]):
                words.extend(c)
            if progress:
                log(f"[decompress] {min(i + self.decode_batch, len(full))}/{len(full)} chunks")
        for c in tail:
            words.extend(self._decode_chunks([c])[0])
        return words

    def flops_per_word(self):
        self._ensure_vocab()
        return lstm_lm_fwd_flops_per_token(self.V, self.meta["emb"], self.meta["hid"])

    def describe(self):
        return {"kind": "learned" if self.scheme == "subseq" else "rule", **self._ctor}
