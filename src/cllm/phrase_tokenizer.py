"""Hybrid tokenizer: multi-word phrases + whole words + sub-word fallback.

Vocabulary (ids in this order): `<unk>`, `<eos>`, fallback pieces, words, phrases.

  phrase    a frequent run of 2..max_n words ("of the", "one of the") -> ONE token
  word      one of the `words` most frequent words -> one token
  fallback  every other word is spelled with WordPiece-style pieces ("zoo" "##logy"); the pieces are
            learned on the rare words only and always include every single character, so any
            word of the training alphabet can be encoded (nothing becomes <unk>)

Encoding is greedy longest-match over the word sequence (phrases first, then the word, then fallback
pieces). Only a fraction of all multi-word combinations get a token: candidate n-grams are counted
(Apriori-style, only over in-vocabulary words), ranked by count * (n - 1) = tokens saved if used, and a
pool of the best `pool_factor * phrases` is then pruned to the `phrases` that the greedy encoder
actually uses most on a slice of the training text (overlapping candidates cannibalise each other, so
real usage is the honest score). Decoding joins tokens with spaces; `##` pieces attach to the previous
token. Lines are encoded independently, so a phrase never spans a line boundary.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from .utils import log

SPECIALS = ["<unk>", "<eos>"]
CONT = "##"
K_SPECIAL, K_FB_INIT, K_FB_CONT, K_WORD, K_PHRASE = range(5)
KIND_NAMES = ["special", "fallback_init", "fallback_cont", "word", "phrase"]


class _Enc:
    __slots__ = ("ids",)

    def __init__(self, ids):
        self.ids = ids


class PhraseTokenizer:
    def __init__(self, fallback, words, phrases, max_n):
        self.fallback, self.words, self.max_n = list(fallback), list(words), int(max_n)
        self.phrases = [tuple(p) for p in phrases]
        strings, kind, nwords = list(SPECIALS), [K_SPECIAL] * len(SPECIALS), [0] * len(SPECIALS)
        self.fb_init, self.fb_cont, self.word_id, self.phrase_id = {}, {}, {}, {}
        for p in self.fallback:
            if p.startswith(CONT):
                self.fb_cont[p[len(CONT):]] = len(strings)
                strings.append(p[len(CONT):])
                kind.append(K_FB_CONT)
            else:
                self.fb_init[p] = len(strings)
                strings.append(p)
                kind.append(K_FB_INIT)
            nwords.append(0)
        for w in self.words:
            self.word_id[w] = len(strings)
            strings.append(w)
            kind.append(K_WORD)
            nwords.append(1)
        for p in self.phrases:
            self.phrase_id[p] = len(strings)
            strings.append(" ".join(p))
            kind.append(K_PHRASE)
            nwords.append(len(p))
        self.strings = strings
        self.kind = np.array(kind, dtype=np.int8)
        self.nwords = np.array(nwords, dtype=np.int32)
        self._starters = {p[0] for p in self.phrases}
        self._maxlen = max([len(p) - (len(CONT) if p.startswith(CONT) else 0) for p in self.fallback] + [1])
        self._fb_cache = {}

    # ---------------- encoding / decoding ----------------
    def _fallback(self, w):
        hit = self._fb_cache.get(w)
        if hit is not None:
            return hit
        out, i, n = [], 0, len(w)
        while i < n:
            table = self.fb_init if i == 0 else self.fb_cont
            j = min(n, i + self._maxlen)
            pid = None
            while j > i:
                pid = table.get(w[i:j])
                if pid is not None:
                    break
                j -= 1
            if pid is None:
                out = [0]                                    # a character outside the alphabet: whole word -> <unk>
                break
            out.append(pid)
            i = j
        self._fb_cache[w] = out
        return out

    def _encode(self, line):
        words = line.split()
        n_w, i, out = len(words), 0, []
        phrase_id, word_id, starters, max_n = self.phrase_id, self.word_id, self._starters, self.max_n
        while i < n_w:
            w = words[i]
            if w in starters:
                for n in range(min(max_n, n_w - i), 1, -1):
                    pid = phrase_id.get(tuple(words[i:i + n]))
                    if pid is not None:
                        out.append(pid)
                        i += n
                        break
                else:
                    pid = None
                if pid is not None:
                    continue
            wid = word_id.get(w)
            if wid is not None:
                out.append(wid)
            else:
                out.extend(self._fallback(w))
            i += 1
        return out

    def encode(self, text):
        return _Enc(self._encode(text))

    def encode_batch(self, lines):
        return [_Enc(self._encode(line)) for line in lines]

    def decode(self, ids, skip_special_tokens=False):
        parts = []
        for i in ids:
            i = int(i)
            k = self.kind[i]
            if k == K_SPECIAL:
                if not skip_special_tokens:
                    parts.append(self.strings[i])
            elif k == K_FB_CONT:
                if parts:
                    parts[-1] += self.strings[i]
                else:
                    parts.append(self.strings[i])
            else:
                parts.append(self.strings[i])
        return " ".join(parts)

    # ---------------- HF-Tokenizer-like API used by the pipeline ----------------
    def get_vocab_size(self):
        return len(self.strings)

    def get_vocab(self):
        """display string -> id (fallback initial pieces are shown as `~piece` so they cannot collide with words)."""
        out = {}
        for i, s in enumerate(self.strings):
            k = self.kind[i]
            out[("~" + s) if k == K_FB_INIT else (CONT + s) if k == K_FB_CONT else s] = i
        return out

    def save(self, path):
        Path(path).write_text(json.dumps({"phrase_tokenizer": 1, "max_n": self.max_n, "fallback": self.fallback,
                                          "words": self.words, "phrases": [list(p) for p in self.phrases]}))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["fallback"], d["words"], d["phrases"], d["max_n"])

    def usage(self, ids):
        """Where the tokens of an id stream come from, and how many words one token covers on average."""
        ids = np.asarray(ids)
        counts = np.bincount(self.kind[ids], minlength=5) / max(len(ids), 1)
        words_covered = float(self.nwords[ids].sum())
        return {**{f"token_frac_{n}": float(c) for n, c in zip(KIND_NAMES, counts)},
                "phrase_words_per_token": float(self.nwords[ids][self.kind[ids] == K_PHRASE].mean())
                if (self.kind[ids] == K_PHRASE).any() else 0.0,
                "n_words_vocab": len(self.words), "n_phrases_vocab": len(self.phrases),
                "n_fallback_pieces": len(self.fallback), "_words_covered": words_covered}

    # ---------------- training ----------------
    @classmethod
    def fit(cls, lines, words=16000, phrases=16000, max_n=4, fallback=1000, min_count=20, pool_factor=2.0,
            refine_words=3_000_000, verbose=True):
        say = log if verbose else (lambda *_: None)
        cnt = Counter()
        for line in lines:
            cnt.update(line.split())
        word_list = [w for w, _ in cnt.most_common(words)]
        rank = {w: i for i, w in enumerate(word_list)}
        say(f"[phrase] {len(cnt):,} word types; word tokens: top {len(word_list):,} "
            f"(cover {sum(cnt[w] for w in word_list) / sum(cnt.values()):.2%} of the text)")

        # ---- sub-word fallback, learned on the words that did not get their own token ----
        alphabet = sorted({c for w in cnt for c in w})
        pieces = set(alphabet) | {CONT + c for c in alphabet}

        def rare_lines():
            for line in lines:
                r = [w for w in line.split() if w not in rank]
                if r:
                    yield " ".join(r)

        if fallback > 2 * len(alphabet) and len(word_list) < len(cnt):
            wp = Tokenizer(models.WordPiece(unk_token="<unk>", max_input_chars_per_word=100))
            wp.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
            wp.train_from_iterator(rare_lines(), trainers.WordPieceTrainer(
                vocab_size=fallback, special_tokens=["<unk>"], continuing_subword_prefix=CONT,
                limit_alphabet=len(alphabet) + 10, show_progress=False))
            pieces |= {p for p in wp.get_vocab() if p != "<unk>"}
        fb = sorted(pieces)
        say(f"[phrase] fallback: {len(fb)} pieces ({len(alphabet)} characters)")

        # ---- multi-word candidates ----
        final = []
        if phrases > 0 and max_n >= 2:
            V = len(word_list)
            if V ** max_n >= 2 ** 62:
                raise ValueError(f"words={V} ** max_n={max_n} overflows the int64 n-gram keys")

            def id_stream():
                for line in lines:
                    for w in line.split():
                        yield rank.get(w, -1)
                    yield -1                                  # phrases never span lines
            ids = np.fromiter(id_stream(), dtype=np.int32)
            cand_n, cand_key, cand_cnt = [], [], []
            for n in range(2, max_n + 1):
                m = len(ids) - n + 1
                keys = np.zeros(m, dtype=np.int64)
                valid = np.ones(m, dtype=bool)
                for j in range(n):
                    col = ids[j:j + m]
                    valid &= col >= 0
                    keys = keys * V + np.maximum(col, 0)
                uk, uc = np.unique(keys[valid], return_counts=True)
                sel = uc >= min_count
                cand_n.append(np.full(sel.sum(), n)); cand_key.append(uk[sel]); cand_cnt.append(uc[sel])
                say(f"[phrase] {n}-grams: {int(sel.sum()):,} with count >= {min_count}")
            del ids
            cn, ck, cc = np.concatenate(cand_n), np.concatenate(cand_key), np.concatenate(cand_cnt)
            order = np.argsort(-(cc * (cn - 1)), kind="stable")[:int(pool_factor * phrases)]

            def to_phrase(key, n):
                out = []
                for _ in range(n):
                    out.append(word_list[int(key % V)])
                    key //= V
                return tuple(reversed(out))
            pool = [to_phrase(int(ck[i]), int(cn[i])) for i in order]
            say(f"[phrase] candidate pool {len(pool):,} phrases; pruning to {phrases:,} by real greedy usage")
            # ---- keep the phrases that are actually used (candidates overlap and cannibalise each other) ----
            sub, nw = [], 0
            for line in lines:
                sub.append(line)
                nw += line.count(" ") + 1
                if nw >= refine_words:
                    break
            probe = cls([], word_list, pool, max_n)
            used = np.zeros(len(probe.strings), dtype=np.int64)
            for e in probe.encode_batch(sub):
                np.add.at(used, np.asarray(e.ids, dtype=np.int64), 1)
            base = len(probe.strings) - len(pool)
            gain = used[base:] * np.array([len(p) - 1 for p in pool])
            keep = np.argsort(-gain, kind="stable")[:phrases]
            final = [pool[i] for i in keep if gain[i] > 0]
            say(f"[phrase] kept {len(final):,} phrases ({len(pool) - len(final):,} pruned); "
                f"top: {[' '.join(p) for p in final[:8]]}")
        return cls(fb, word_list, final, max_n)
