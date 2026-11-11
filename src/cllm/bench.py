"""Inference benchmarks (generation latency, memory, decompressor overhead) and sample generation."""
import time

import torch

from .tokenizer import decode_ids
from .utils import peak_memory_mb, reset_peak_memory, sync


@torch.no_grad()
def _timed_generate(model, prompt, n_new, device, runs=3):
    best = float("inf")
    for _ in range(runs):
        sync(device)
        t0 = time.time()
        model.generate(prompt, max_new_tokens=n_new, min_new_tokens=n_new, do_sample=False,
                       pad_token_id=1, eos_token_id=None)
        sync(device)
        best = min(best, time.time() - t0)
    return best


def inference_bench(model, comp, test_ids, test_comp_words, orig_chars, orig_words, ctx, device,
                    prompt_tokens=32, gen_tokens=128, overhead_words=4096):
    """Latency per generated token (KV cache, batch 1), peak memory, and the compressor /
    decompressor cost per 1k original characters, so compressed pipelines are charged for them."""
    gen_tokens = min(gen_tokens, ctx - prompt_tokens - 1)
    prompt = torch.from_numpy(test_ids[:prompt_tokens].astype("int64"))[None].to(device)
    _timed_generate(model, prompt, 4, device, runs=1)                      # warm-up
    reset_peak_memory(device)
    t1 = _timed_generate(model, prompt, 1, device)
    tn = _timed_generate(model, prompt, gen_tokens, device)
    per_token = (tn - t1) / max(gen_tokens - 1, 1)
    peak = peak_memory_mb(device)
    cpt = orig_chars / len(test_ids)                                       # original chars per token
    n = min(overhead_words, len(test_comp_words))
    n -= n % 256 if n >= 256 else 0
    words = test_comp_words[:n]
    t0 = time.time()
    dec = comp.decompress(words)
    t_dec = time.time() - t0
    orig_chars_n = sum(len(w) for w in dec) + len(dec)
    t0 = time.time()
    comp.compress(dec)
    t_comp = time.time() - t0
    k = 1000 / max(orig_chars_n, 1)
    gen_ms_1k = per_token * 1000 * (1000 / cpt)
    return {
        "prefill_ms_prompt": t1 * 1000,
        "decode_ms_per_token": per_token * 1000,
        "decode_tokens_per_s": 1 / max(per_token, 1e-9),
        "orig_chars_per_token": cpt,
        "orig_chars_per_s_generated": cpt / max(per_token, 1e-9),
        "peak_infer_mem_mb": peak,
        "decompress_ms_per_1k_orig_chars": t_dec * 1000 * k,
        "compress_ms_per_1k_orig_chars": t_comp * 1000 * k,
        "generate_ms_per_1k_orig_chars": gen_ms_1k,
        "end_to_end_ms_per_1k_orig_chars": gen_ms_1k + t_dec * 1000 * k,
    }


@torch.no_grad()
def generate_samples(model, tok, comp, test_comp_words, test_words, device, n_samples=3, prompt_words=48,
                     gen_tokens=160, ctx=512, chunk=256, seed=0):
    """Prompt the LLM with compressed test text, sample a continuation, and decompress it."""
    torch.manual_seed(seed)
    out = []
    for s in range(n_samples):
        off = s * chunk
        if off + prompt_words > len(test_comp_words):
            break
        prompt_toks = test_comp_words[off:off + prompt_words]
        ids = tok.encode(" ".join(prompt_toks)).ids
        n_new = max(1, min(gen_tokens, ctx - len(ids) - 1))
        x = torch.tensor(ids, dtype=torch.long)[None].to(device)
        y = model.generate(x, max_new_tokens=n_new, do_sample=True, top_k=40, temperature=0.8,
                           pad_token_id=1, eos_token_id=None)[0].tolist()
        text = decode_ids(tok, y)
        words = text.split()
        if len(words) > prompt_words:
            words = words[:-1]                       # last word may be cut mid-token
        restored = comp.decompress(words)
        out.append({
            "prompt_compressed": " ".join(prompt_toks),
            "continuation_compressed": " ".join(words[prompt_words:]),
            "prompt_original": " ".join(test_words[off:off + prompt_words]),
            "prompt_restored": " ".join(restored[:prompt_words]),
            "continuation_restored": " ".join(restored[prompt_words:]),
        })
    return out
