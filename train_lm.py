"""Train the LSTM LM for a fixed wall-clock budget (cosine LR over time)."""
import sys, time, json, math
import numpy as np
import torch
import torch.nn.functional as F
from lm import *

BUDGET_MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 35
BATCH = 16

dev = "mps" if torch.backends.mps.is_available() else "cpu"
train, val, _ = load_words()
vocab = build_vocab(train)
w2i = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
json.dump(vocab, open("models/vocab.json", "w"))
tr_in, tr_tg = make_chunks(to_ids(train, w2i), V)
va_in, va_tg = make_chunks(to_ids(val, w2i), V)
print(f"V={V} train chunks {len(tr_in)} val chunks {len(va_in)} device {dev}", flush=True)

model = LM(V).to(dev)
print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M", flush=True)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
LR_MAX = 2e-3


def evaluate():
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(va_in), 32):
            x = torch.from_numpy(va_in[i:i + 32]).to(dev)
            y = torch.from_numpy(va_tg[i:i + 32]).to(dev)
            lg, _ = model(x)
            tot += F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction="sum").item()
            n += y.numel()
    model.train()
    return tot / n


t0, step, budget = time.time(), 0, BUDGET_MIN * 60
rng = np.random.default_rng(0)
done = False
while not done:
    perm = rng.permutation(len(tr_in))
    for i in range(0, len(perm) - BATCH, BATCH):
        frac = (time.time() - t0) / budget
        if frac >= 1:
            done = True
            break
        lr = LR_MAX * (min(1, step / 200)) * (0.5 * (1 + math.cos(math.pi * frac)) * 0.97 + 0.03)
        for g in opt.param_groups:
            g["lr"] = lr
        idx = perm[i:i + BATCH]
        x = torch.from_numpy(tr_in[idx]).to(dev)
        y = torch.from_numpy(tr_tg[idx]).to(dev)
        lg, _ = model(x)
        loss = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 200 == 0:
            print(f"step {step} {frac*100:.0f}% loss {loss.item():.3f} lr {lr:.5f} t {time.time()-t0:.0f}s", flush=True)
        if step % 1500 == 0:
            vl = evaluate()
            print(f"  val nll {vl:.3f} ppl {math.exp(vl):.1f} ({vl/math.log(2):.2f} bits/word)", flush=True)
            torch.save(model.state_dict(), "models/lm.pt")
vl = evaluate()
print(f"FINAL val nll {vl:.3f} ppl {math.exp(vl):.1f} ({vl/math.log(2):.2f} bits/word)", flush=True)
torch.save(model.state_dict(), "models/lm.pt")
