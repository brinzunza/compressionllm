"""Word-level LSTM language model: the shared prior behind the letter-dropping decoders.

Context is reset every `chunk` words, so independent chunks can be encoded (teacher forced) and
decoded (stepwise) in parallel. Token 0 is <unk>; input id V is a beginning-of-chunk marker.
"""
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .flops import lstm_lm_fwd_flops_per_token
from .utils import get_device, log, peak_memory_mb, reset_peak_memory, set_seed, sync


class WordLM(nn.Module):
    def __init__(self, V, emb=256, hid=512, dropout=0.2):
        super().__init__()
        self.V = V
        self.emb = nn.Embedding(V + 1, emb)
        self.rnn = nn.LSTM(emb, hid, batch_first=True)
        self.proj = nn.Linear(hid, emb)
        self.bias = nn.Parameter(torch.zeros(V))
        self.drop = nn.Dropout(dropout)

    def logits_from_h(self, h):
        return self.proj(self.drop(h)) @ self.emb.weight[: self.V].T + self.bias

    def forward(self, inp, state=None):
        h, state = self.rnn(self.drop(self.emb(inp)), state)
        return self.logits_from_h(h), state


def build_vocab(train_words, v_max, max_len):
    cnt = Counter(train_words)
    top = [w for w, _ in cnt.most_common() if len(w) <= max_len][:v_max]
    return ["<unk>"] + top


def to_ids(words, w2i):
    return np.fromiter((w2i.get(w, 0) for w in words), dtype=np.int64, count=len(words))


def make_chunks(ids, V, chunk):
    n = len(ids) // chunk
    tgt = ids[: n * chunk].reshape(n, chunk)
    inp = np.concatenate([np.full((n, 1), V, dtype=np.int64), tgt[:, :-1]], axis=1)
    return inp, tgt


def train_word_lm(train_words, val_words, cfg, out_dir, device=None):
    """Train for `steps` optimizer steps or `budget_minutes` of wall clock, whichever is configured."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = device or get_device(cfg.get("device", "auto"))
    set_seed(cfg.get("seed", 0))
    chunk, batch = cfg.get("chunk", 256), cfg.get("batch", 16)
    emb, hid = cfg.get("emb", 256), cfg.get("hid", 512)
    max_len = cfg.get("max_word_len", 12)
    lr_max, steps_cfg, minutes = cfg.get("lr", 2e-3), cfg.get("steps"), cfg.get("budget_minutes", 20)

    vocab = build_vocab(train_words, cfg.get("vocab_size", 20000), max_len)
    w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    tr_in, tr_tg = make_chunks(to_ids(train_words, w2i), V, chunk)
    va_in, va_tg = make_chunks(to_ids(val_words, w2i), V, chunk)
    model = WordLM(V, emb, hid, cfg.get("dropout", 0.2)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr_max)
    log(f"[word-lm] V={V} chunks train/val {len(tr_in)}/{len(va_in)} params {n_params/1e6:.2f}M device {device}")

    def evaluate():
        if len(va_in) == 0:
            return float("nan")
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(va_in), 32):
                x = torch.from_numpy(va_in[i:i + 32]).to(device)
                y = torch.from_numpy(va_tg[i:i + 32]).to(device)
                lg, _ = model(x)
                tot += F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction="sum").item()
                n += y.numel()
        model.train()
        return tot / n

    reset_peak_memory(device)
    rng = np.random.default_rng(cfg.get("seed", 0))
    t0, step, done = time.time(), 0, False
    eval_every = cfg.get("eval_every", 1500)
    while not done:
        perm = rng.permutation(len(tr_in))
        for i in range(0, len(perm) - batch + 1, batch):
            frac = step / steps_cfg if steps_cfg else (time.time() - t0) / (minutes * 60)
            if frac >= 1:
                done = True
                break
            lr = lr_max * min(1, (step + 1) / 200) * (0.5 * (1 + math.cos(math.pi * frac)) * 0.97 + 0.03)
            for g in opt.param_groups:
                g["lr"] = lr
            idx = perm[i:i + batch]
            x = torch.from_numpy(tr_in[idx]).to(device)
            y = torch.from_numpy(tr_tg[idx]).to(device)
            lg, _ = model(x)
            loss = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 200 == 0:
                log(f"[word-lm] step {step} {frac*100:.0f}% loss {loss.item():.3f} lr {lr:.5f}")
            if step % eval_every == 0:
                vl = evaluate()
                log(f"[word-lm]   val nll {vl:.3f} ppl {math.exp(vl):.1f}")
        if len(perm) < batch:
            break
    sync(device)
    secs = time.time() - t0
    vl = evaluate()
    tokens = step * batch * chunk
    fwd = lstm_lm_fwd_flops_per_token(V, emb, hid)
    stats = {
        "vocab": V, "params": n_params, "steps": step, "tokens_seen": tokens, "seconds": secs,
        "val_nll": vl, "val_ppl": math.exp(vl) if vl == vl else None,
        "train_flops": 3 * fwd * tokens, "fwd_flops_per_word": fwd, "peak_mem_mb": peak_memory_mb(device),
    }
    log(f"[word-lm] done: {step} steps in {secs:.0f}s, val ppl {stats['val_ppl']}")
    torch.save(model.state_dict(), out / "lm.pt")
    (out / "vocab.json").write_text(json.dumps(vocab))
    (out / "meta.json").write_text(json.dumps({"V": V, "emb": emb, "hid": hid, "chunk": chunk, "max_word_len": max_len}))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    (out / "DONE").write_text("ok")
    return stats


def load_word_lm(lm_dir, device="cpu"):
    d = Path(lm_dir)
    meta = json.loads((d / "meta.json").read_text())
    vocab = json.loads((d / "vocab.json").read_text())
    model = WordLM(meta["V"], meta["emb"], meta["hid"])
    model.load_state_dict(torch.load(d / "lm.pt", map_location="cpu", weights_only=True))
    return model.to(device).eval(), vocab, meta
