"""Small GPT-2-style language model trained on a token-id stream, with an equal-budget trainer.

The headline metric is bits per ORIGINAL character on held-out text (nats over tokens / ln 2 /
original characters), which is comparable across raw and compressed streams as long as the
compressor is lossless. Budgets can be matched by tokens (~equal FLOPs), by original characters
seen, or by FLOPs.
"""
import datetime
import json
import math
import socket
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from .flops import gpt_param_counts, gpt_train_flops_per_token
from .utils import get_device, log, peak_memory_mb, reset_peak_memory, set_seed, sync


def build_model(cfg, vocab_size):
    gcfg = GPT2Config(
        vocab_size=vocab_size, n_positions=cfg["ctx"], n_embd=cfg["n_embd"], n_layer=cfg["n_layer"],
        n_head=cfg["n_head"], resid_pdrop=cfg.get("dropout", 0.0), embd_pdrop=cfg.get("dropout", 0.0),
        attn_pdrop=cfg.get("dropout", 0.0), bos_token_id=1, eos_token_id=1,
    )
    try:
        return GPT2LMHeadModel._from_config(gcfg, attn_implementation="sdpa")
    except Exception:
        return GPT2LMHeadModel(gcfg)


def autocast_ctx(device, precision="auto"):
    d = torch.device(device)
    if d.type != "cuda" or precision == "fp32":
        return torch.autocast("cpu", enabled=False), None
    if precision in ("auto", "bf16") and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16), None
    return torch.autocast("cuda", dtype=torch.float16), torch.cuda.amp.GradScaler()


def budget_to_tokens(budget, flops_per_token, chars_per_token):
    mode, value = budget["mode"], budget["value"]
    if mode == "tokens":
        return int(value)
    if mode == "orig_chars":
        return int(value / chars_per_token)
    if mode == "flops":
        return int(value / flops_per_token)
    raise ValueError(f"unknown budget mode {mode}")


@torch.no_grad()
def eval_nll(model, ids, ctx, device, batch=8, stride=None, max_tokens=None, precision="auto"):
    """Sliding-window NLL where every scored token has at least ctx-stride tokens of context
    (except inside the first window). Returns total nats and number of scored tokens."""
    ids = np.asarray(ids[:max_tokens] if max_tokens else ids, dtype=np.int64)
    n = len(ids)
    if n < 2:
        return 0.0, 0
    ctx = min(ctx, n)
    stride = stride or max(ctx // 2, 1)
    starts = list(range(0, n - ctx + 1, stride))
    if starts[-1] != n - ctx:
        starts.append(n - ctx)
    rmin = [1] + [max(1, starts[k - 1] + ctx - starts[k]) for k in range(1, len(starts))]
    model.eval()
    total, count = 0.0, 0
    ac, _ = autocast_ctx(device, precision)
    for i in range(0, len(starts), batch):
        ss = starts[i:i + batch]
        x = torch.from_numpy(np.stack([ids[s:s + ctx] for s in ss])).to(device)
        with ac:
            logits = model(input_ids=x).logits
        nll = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1),
                              reduction="none").view(len(ss), ctx - 1)
        for b in range(len(ss)):
            r = rmin[i + b]
            total += nll[b, r - 1:].sum().item()
            count += ctx - r
    return total, count


def bits_per_char(nats, n_scored, chars_per_token):
    """Total NLL (nats) over n_scored tokens -> bits per ORIGINAL character."""
    return nats / math.log(2) / max(n_scored * chars_per_token, 1e-9)


def resolve_world(cfg, device):
    """Number of training processes: `llm.gpus` = 1 (default), N, or "all" (every visible CUDA GPU).
    On CPU, N > 1 runs N gloo processes (useful for testing DDP without GPUs)."""
    g = cfg.get("gpus", 1)
    if torch.device(device).type != "cuda":
        return 1 if g in ("all", None) else int(g)
    n = torch.cuda.device_count()
    world = n if g == "all" else int(g)
    if world > n:
        raise ValueError(f"llm.gpus={world} but only {n} CUDA device(s) are visible")
    return max(world, 1)


def train_llm(cfg, ids_train, ids_val, vocab_size, train_cpt, val_cpt, out_dir, device=None):
    """Train with a matched budget. `*_cpt` = original characters per token for the train/val streams.

    `llm.batch_size` is the GLOBAL batch (sequences per optimizer micro-step across all GPUs), so the
    optimization is the same whatever `llm.gpus` is; each rank gets batch_size // gpus sequences.
    With gpus > 1 this spawns one DDP process per GPU and returns the rank-0 model and stats."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = device or get_device(cfg.get("device", "auto"))
    world = resolve_world(cfg, device)
    if world == 1:
        return _train(0, 1, cfg, np.asarray(ids_train), ids_val, vocab_size, train_cpt, val_cpt, out, device)
    import torch.multiprocessing as mp
    if cfg["batch_size"] % world:
        raise ValueError(f"llm.batch_size={cfg['batch_size']} must be divisible by llm.gpus={world}")
    paths = {}
    for k, v in (("train", ids_train), ("val", ids_val)):
        paths[k] = out / f"_ids_{k}.npy"
        np.save(paths[k], np.asarray(v))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    try:
        mp.spawn(_ddp_worker, nprocs=world, join=True,
                 args=(world, f"tcp://127.0.0.1:{port}", cfg, paths, vocab_size, train_cpt, val_cpt, out,
                       torch.device(device).type))
    finally:
        for p in paths.values():
            p.unlink(missing_ok=True)
    stats = json.loads((out / "_train_stats.json").read_text())
    (out / "_train_stats.json").unlink()
    return load_llm(out, device), stats


def _ddp_worker(rank, world, init_method, cfg, paths, vocab_size, train_cpt, val_cpt, out, dev_type):
    import torch.distributed as dist
    backend = "nccl" if dev_type == "cuda" else "gloo"
    if dev_type == "cuda":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, init_method=init_method, rank=rank, world_size=world,
                            timeout=datetime.timedelta(minutes=30))
    device = torch.device(f"cuda:{rank}") if dev_type == "cuda" else torch.device("cpu")
    try:
        ids_train = np.load(paths["train"], mmap_mode="r")
        ids_val = np.load(paths["val"], mmap_mode="r")
        _train(rank, world, cfg, ids_train, ids_val, vocab_size, train_cpt, val_cpt, Path(out), device)
    finally:
        dist.destroy_process_group()


def _train(rank, world, cfg, ids_train, ids_val, vocab_size, train_cpt, val_cpt, out, device):
    from contextlib import nullcontext
    main = rank == 0
    set_seed(cfg.get("seed", 0))             # identical init on every rank (DDP also broadcasts)
    ctx, bs, accum = cfg["ctx"], cfg["batch_size"], cfg.get("grad_accum", 1)
    bs_local = bs // world
    model = build_model(cfg, vocab_size).to(device)
    dmodel = model
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        dmodel = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    counts = gpt_param_counts(cfg["n_layer"], cfg["n_embd"], ctx, vocab_size)
    n_params = sum(p.numel() for p in model.parameters())
    fpt = gpt_train_flops_per_token(cfg["n_layer"], cfg["n_embd"], ctx, vocab_size)
    tokens_per_step = bs * ctx * accum                       # global
    budget_tokens = budget_to_tokens(cfg["budget"], fpt, train_cpt)
    steps = max(1, budget_tokens // tokens_per_step)
    if main:
        log(f"[llm] params {n_params/1e6:.2f}M (analytic non-emb {counts['non_embedding']/1e6:.2f}M) vocab {vocab_size} "
            f"| {steps} steps x {tokens_per_step} tokens = {steps*tokens_per_step/1e6:.1f}M tokens "
            f"({steps*tokens_per_step*train_cpt/1e6:.1f}M original chars) device {device} x{world} gpu(s), "
            f"{bs_local} seq/gpu/micro-step")

    decay = [p for n_, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n_, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.get("weight_decay", 0.1)},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg["lr"], betas=(0.9, 0.95), fused=device.type == "cuda")
    ac, scaler = autocast_ctx(device, cfg.get("precision", "auto"))
    data = torch.from_numpy(np.asarray(ids_train, dtype=np.int64)).to(device)
    n_data = len(data)
    if n_data <= ctx + 1:
        raise ValueError("training stream shorter than context window")
    warm = cfg.get("warmup_steps", max(1, steps // 50))
    lr_min = cfg["lr"] * cfg.get("min_lr_frac", 0.1)
    eval_every = cfg.get("eval_every", max(steps // 10, 1))
    eval_tokens = cfg.get("eval_tokens", 500_000)
    curve = []
    model.train()
    reset_peak_memory(device)
    sync(device)
    t0 = time.time()
    gen = torch.Generator(device="cpu").manual_seed(cfg.get("seed", 0) + 1000 * rank)   # different data per rank
    last_loss = float("nan")
    for step in range(1, steps + 1):
        lr = cfg["lr"] * step / warm if step <= warm else lr_min + 0.5 * (cfg["lr"] - lr_min) * (
            1 + math.cos(math.pi * (step - warm) / max(steps - warm, 1)))
        for g in opt.param_groups:
            g["lr"] = lr
        for j in range(accum):
            ix = torch.randint(0, n_data - ctx - 1, (bs_local,), generator=gen).to(device)
            x = torch.stack([data[i:i + ctx] for i in ix.tolist()])
            skip_sync = dmodel.no_sync() if (world > 1 and j < accum - 1) else nullcontext()
            with skip_sync:
                with ac:
                    loss = dmodel(input_ids=x, labels=x).loss / accum
                (scaler.scale(loss) if scaler else loss).backward()
        if scaler:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 20 == 0 or step == steps:
            last_loss = loss.item() * accum                  # rank-0 local loss (other ranks see other batches)
        if step % eval_every == 0 or step == steps:
            sync(device)
            if main:
                elapsed = time.time() - t0
                nats, cnt = eval_nll(model, ids_val, ctx, device, max_tokens=eval_tokens,
                                     precision=cfg.get("precision", "auto"))
                model.train()
                pt = {"step": step, "tokens": step * tokens_per_step,
                      "orig_chars": step * tokens_per_step * train_cpt, "flops": step * tokens_per_step * fpt,
                      "elapsed_s": elapsed, "train_loss": last_loss, "val_bpc": bits_per_char(nats, cnt, val_cpt)}
                curve.append(pt)
                log(f"[llm] step {step}/{steps} loss {last_loss:.3f} val_bpc {pt['val_bpc']:.4f} "
                    f"{step*tokens_per_step/elapsed/1e3:.0f}k tok/s")
            if world > 1:
                import torch.distributed as dist
                dist.barrier()                               # other ranks wait while rank 0 evaluates
    sync(device)
    secs = time.time() - t0
    stats = None
    if main:
        model.save_pretrained(out / "model")
        stats = {
            "params_total": n_params, "params_non_embedding_analytic": counts["non_embedding"],
            "vocab_size": vocab_size, "steps": steps, "tokens_seen": steps * tokens_per_step,
            "orig_chars_seen": steps * tokens_per_step * train_cpt, "train_flops": steps * tokens_per_step * fpt,
            "train_seconds": secs, "train_tokens_per_s": steps * tokens_per_step / secs,
            "peak_train_mem_mb": peak_memory_mb(device),     # rank 0
            "final_train_loss": last_loss, "curve": curve, "world_size": world, "global_batch": bs,
        }
        if world > 1:
            (out / "_train_stats.json").write_text(json.dumps(stats))
    return model, stats


def load_llm(run_dir, device):
    return GPT2LMHeadModel.from_pretrained(Path(run_dir) / "model").to(device).eval()
