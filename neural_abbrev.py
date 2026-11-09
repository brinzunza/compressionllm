"""Neural abbreviation codec.

Decoder: LSTM LM gives logits over the vocabulary given the previous decoded
words (context reset every CHUNK words); the abbreviation masks the vocabulary
to words containing it as a subsequence (same first letter); argmax wins.
Encoder: for every word, the shortest abbreviation whose masked argmax is the
true word with a logit margin >= TAU over every competitor (the margin makes
the scheme robust to float differences between batched encoding and stepwise
decoding). Otherwise the word is escaped as '#word'.
"""
import sys, time, math, json, lzma, zlib
from itertools import combinations
import numpy as np
import torch
from lm import *

TAU = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
N_CHUNKS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
CHAR_BITS = math.log2(27)        # text8 alphabet, for spelling unk words in the ideal-AC estimate

torch.set_num_threads(8)
vocab = json.load(open("models/vocab.json"))
w2i = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
model = LM(V)
model.load_state_dict(torch.load("models/lm.pt", map_location="cpu", weights_only=True))
model.eval()

_, _, test_all = load_words()
test = test_all[: N_CHUNKS * CHUNK]
tgt = to_ids(test, w2i).reshape(N_CHUNKS, CHUNK)
inp = np.concatenate([np.full((N_CHUNKS, 1), V, dtype=np.int64), tgt[:, :-1]], axis=1)

t0 = time.time()
inv = {}
for wid in range(1, V):
    w = vocab[wid]
    for k in range(len(w)):
        for comb in combinations(range(1, len(w)), k):
            inv.setdefault(w[0] + "".join(w[i] for i in comb), []).append(wid)
inv = {a: np.array(l, dtype=np.int32) for a, l in inv.items()}
print(f"index {len(inv):,} abbreviations ({time.time()-t0:.0f}s)", flush=True)

# ---------------- encode (teacher-forced, one chunk at a time) ----------------
toks = [[] for _ in range(N_CHUNKS)]
nll_bits = 0.0
with torch.no_grad():
    for c in range(N_CHUNKS):
        lg, _ = model(torch.from_numpy(inp[c:c + 1]))
        L = lg[0].double().numpy()
        lse = np.log(np.exp(L - L.max(1, keepdims=True)).sum(1)) + L.max(1)
        L32 = lg[0].numpy()
        for j in range(CHUNK):
            wid = int(tgt[c, j])
            word = test[c * CHUNK + j]
            nll_bits += (lse[j] - L[j, wid]) / math.log(2)
            if wid == 0:
                nll_bits += CHAR_BITS * (len(word) + 1)
                toks[c].append("#" + word)
                continue
            row, rw, tok = L32[j], L32[j, wid], None
            for k in range(len(word)):
                for comb in combinations(range(1, len(word)), k):
                    a = word[0] + "".join(word[i] for i in comb)
                    ids = inv[a]
                    if len(ids) > 1:
                        v = row[ids].copy()
                        v[ids == wid] = -np.inf
                        if v.max() >= rw - TAU:
                            continue
                    tok = a
                    break
                if tok:
                    break
            toks[c].append(tok if tok else "#" + word)
print(f"encoded ({time.time()-t0:.0f}s)", flush=True)

# ---------------- decode (stepwise, all chunks in parallel) ----------------
out = [[] for _ in range(N_CHUNKS)]
prev = np.full(N_CHUNKS, V, dtype=np.int64)
state = None
with torch.no_grad():
    for t in range(CHUNK):
        h, state = model.rnn(model.emb(torch.from_numpy(prev))[:, None, :], state)
        lg = model.logits_from_h(h[:, 0]).numpy()
        for b in range(N_CHUNKS):
            tk = toks[b][t]
            if tk[0] == "#":
                w = tk[1:]
                wid = w2i.get(w, 0)
            else:
                ids = inv[tk]
                wid = int(ids[np.argmax(lg[b][ids])])
                w = vocab[wid]
            out[b].append(w)
            prev[b] = wid
print(f"decoded ({time.time()-t0:.0f}s)", flush=True)

flat = [t for c in toks for t in c]
back = [w for c in out for w in c]
bad = sum(a != b for a, b in zip(back, test))
esc = sum(t[0] == "#" for t in flat) / len(flat)
oov = float((tgt == 0).mean())
orig, comp = " ".join(test), " ".join(flat)
print(f"test words {len(test):,}  TAU {TAU}  roundtrip mismatches {bad} ({'OK' if bad == 0 else 'FAILED'})")
print(f"escapes {esc:.1%} (OOV/too long {oov:.1%})")
print(f"mean chars/word: orig {len(orig)/len(test):.2f}  abbrev {len(comp)/len(test):.2f}")


def sizes(s):
    b = s.encode()
    return len(b), len(zlib.compress(b, 9)), len(lzma.compress(b, preset=9))


so, sc = sizes(orig), sizes(comp)
print(f"{'':10s}{'raw':>10s}{'zlib9':>10s}{'xz9':>10s}")
print(f"{'orig':10s}{so[0]:>10,}{so[1]:>10,}{so[2]:>10,}")
print(f"{'abbrev':10s}{sc[0]:>10,}{sc[1]:>10,}{sc[2]:>10,}")
print(f"ratios abbrev/orig: raw {sc[0]/so[0]:.3f}  zlib {sc[1]/so[1]:.3f}  xz {sc[2]/so[2]:.3f}")
print(f"LM + arithmetic coding (ideal, same LSTM): {nll_bits/8:,.0f} bytes = {nll_bits/so[2]/8:.3f} x xz(orig), {nll_bits/len(test):.2f} bits/word")
open("models/abbrev_stream.txt", "w").write(comp)
