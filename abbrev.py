"""Context-aware abbreviation compression: baseline with an n-gram decoder.

Idea: replace each word by a short subsequence of its letters (first letter
kept). A decoder that sees the previous *decoded* words picks the most likely
vocabulary word consistent with the abbreviation. Because the decoder is
deterministic, the encoder can simulate it and search for the shortest
abbreviation that decodes back to the true word (analysis-by-synthesis).
Words where no abbreviation works are escaped as '#word'.

The decoder is a pluggable function decode(abbrev, prev_word_id) -> word_id,
so a neural LM can replace the n-gram model later.
"""
import sys, time, math, lzma, zlib
from itertools import combinations
from collections import Counter
import numpy as np

V_MAX, MAXLEN, K, LAM = 20000, 12, 24, 0.7
TRAIN_CHARS = 90_000_000


def is_subseq(a, word):
    if a[0] != word[0]:
        return False
    it = iter(word)
    return all(c in it for c in a)


class Model:
    def __init__(self, train_words):
        cnt = Counter(train_words)
        top = [w for w, _ in cnt.most_common() if len(w) <= MAXLEN][:V_MAX]
        self.words = ["<unk>"] + top
        self.id = {w: i for i, w in enumerate(self.words)}
        V = len(self.words)
        ids = np.fromiter((self.id.get(w, 0) for w in train_words), dtype=np.int64, count=len(train_words))
        uni = np.bincount(ids, minlength=V).astype(np.float64)
        tot = uni[1:].sum()
        self.pu = (uni / tot).tolist()
        keys = ids[:-1] * V + ids[1:]
        uk, uc = np.unique(keys, return_counts=True)
        prev, nxt = uk // V, uk % V
        m = nxt != 0
        prev, nxt, uc = prev[m], nxt[m], uc[m]
        order = np.lexsort((-uc, prev))
        prev, nxt, uc = prev[order], nxt[order], uc[order]
        starts = np.searchsorted(prev, np.arange(V + 1))
        self.succ, self.succ_d = {}, {}
        for p in range(V):
            s, e = starts[p], min(starts[p + 1], starts[p] + K)
            if e > s:
                bi = (LAM * uc[s:e] / uni[p]).tolist()
                ws = nxt[s:e].tolist()
                self.succ[p] = list(zip(ws, bi))
                self.succ_d[p] = dict(zip(ws, bi))
        # index: abbreviation -> most frequent (unigram) word containing it
        self.idx = {}
        for wid in sorted(range(1, V), key=lambda i: -uni[i]):
            w = self.words[wid]
            for k in range(len(w)):
                for comb in combinations(range(1, len(w)), k):
                    self.idx.setdefault(w[0] + "".join(w[i] for i in comb), wid)

    def score(self, wid, prev):
        return self.succ_d.get(prev, {}).get(wid, 0.0) + (1 - LAM) * self.pu[wid]

    def decode(self, a, prev):
        best = self.idx.get(a)
        if best is None:
            return 0
        bs = self.score(best, prev)
        for s, bi in self.succ.get(prev, ()):
            if s != best and is_subseq(a, self.words[s]):
                sc = bi + (1 - LAM) * self.pu[s]
                if sc > bs or (sc == bs and s < best):
                    best, bs = s, sc
        return best

    def encode_word(self, w, wid, prev):
        if wid == 0:
            return "#" + w
        L = len(w)
        for k in range(L):
            for comb in combinations(range(1, L), k):
                a = w[0] + "".join(w[i] for i in comb)
                if self.decode(a, prev) == wid:
                    return a
        return "#" + w

    def encode(self, words):
        out, prev = [], 0
        for w in words:
            wid = self.id.get(w, 0)
            out.append(self.encode_word(w, wid, prev))
            prev = wid
        return out

    def decode_all(self, toks):
        out, prev = [], 0
        for t in toks:
            if t[0] == "#":
                w = t[1:]
                wid = self.id.get(w, 0)
            else:
                wid = self.decode(t, prev)
                w = self.words[wid]
            out.append(w)
            prev = wid
        return out


def sizes(s):
    b = s.encode()
    return len(b), len(zlib.compress(b, 9)), len(lzma.compress(b, preset=9))


def main(n_test_words=100_000):
    t0 = time.time()
    text = open("data/text8").read()
    train = text[:TRAIN_CHARS].split()
    test = text[TRAIN_CHARS:].split()[1:n_test_words + 1]
    print(f"train words {len(train):,}  test words {len(test):,}")
    m = Model(train)
    print(f"model built: vocab {len(m.words):,}, index {len(m.idx):,}  ({time.time()-t0:.0f}s)")
    toks = m.encode(test)
    print(f"encoded ({time.time()-t0:.0f}s)")
    back = m.decode_all(toks)
    assert back == test, "ROUNDTRIP FAILED"
    orig, comp = " ".join(test), " ".join(toks)
    esc = sum(t[0] == "#" for t in toks) / len(toks)
    oov = sum(m.id.get(w, 0) == 0 for w in test) / len(test)
    print(f"roundtrip OK. escapes {esc:.1%} (of which OOV/too long {oov:.1%})")
    print(f"mean chars/word: orig {len(orig)/len(test):.2f}  abbrev {len(comp)/len(test):.2f}")
    print(f"{'':10s}{'raw':>10s}{'zlib9':>10s}{'xz9':>10s}")
    so, sc = sizes(orig), sizes(comp)
    print(f"{'orig':10s}{so[0]:>10,}{so[1]:>10,}{so[2]:>10,}")
    print(f"{'abbrev':10s}{sc[0]:>10,}{sc[1]:>10,}{sc[2]:>10,}")
    print(f"raw ratio {sc[0]/so[0]:.3f}  zlib ratio {sc[1]/so[1]:.3f}  xz ratio {sc[2]/so[2]:.3f}  (vs orig-compressed: xz {sc[2]/so[2]:.3f})")
    # rough LM-ideal reference: -log2 score of each in-vocab word, 8 bits/char for escapes
    bits, prev = 0.0, 0
    for w in test:
        wid = m.id.get(w, 0)
        if wid:
            bits += -math.log2(max(m.score(wid, prev), 1e-9))
        else:
            bits += 8 * (len(w) + 1) + 14
        prev = wid
    print(f"LM-ideal (same n-gram, unnormalised) ~ {bits/8:,.0f} bytes  (xz of orig: {so[2]:,})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100_000)
