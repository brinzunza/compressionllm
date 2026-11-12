"""Experiment pipeline. Every stage is lazy and cached by a hash of the config sections it depends on:

    data -> decoder LM -> compressed corpus -> tokenizer -> LLM training -> eval + benchmarks

Artifacts live in `artifacts/<stage>/<hash>/` (shared between experiments that agree on the inputs
of that stage); per-experiment results live in `runs/<name>/`.
"""
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .bench import generate_samples, inference_bench
from .compressors import build_compressor
from .config import stage_hash
from .data import load_words, n_chars
from .evaluate import compression_report
from .flops import gpt_train_flops_per_token
from .llm import bits_per_char, eval_nll, train_llm
from .phrase_tokenizer import PhraseTokenizer
from .tokenizer import (encode_words, load_tokenizer, sweep, tokenizer_report, to_lines, train_tokenizer)
from .utils import get_device, hardware_info, log, set_seed
from .word_lm import train_word_lm

LINE_WORDS = 64


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=float))


def _read_json(path):
    return json.loads(Path(path).read_text())


def _write_tokens(path, tokens):
    Path(path).write_text("\n".join(to_lines(tokens, LINE_WORDS)) + "\n")


def _read_tokens(path):
    return Path(path).read_text().split()


class Experiment:
    def __init__(self, cfg, force=False, until=None):
        self.cfg, self.force, self.until = cfg, force, until
        self.name = cfg["name"]
        self.art = Path(cfg.get("artifacts_dir", "artifacts"))
        self.runs = Path(cfg.get("runs_dir", "runs"))
        self._words = None
        self._comp = None

    # ---------------- helpers ----------------
    @property
    def words(self):
        if self._words is None:
            self._words = load_words(self.cfg["data"])
        return self._words

    def _forced(self, stage):
        return self.force and self.until == stage

    def _cached(self, d, stage):
        if (d / "DONE").exists() and not self._forced(stage):
            return True
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        return False

    def _comp_cfg(self):
        return {k: v for k, v in self.cfg["compressor"].items() if k != "workers"}

    def _lm_hash(self):
        d = self.cfg["data"]
        return stage_hash({k: d[k] for k in ("n_train_words", "n_val_words")}, self.cfg.get("decoder_lm", {}))

    # ---------------- stage: decoder LM ----------------
    def stage_decoder_lm(self):
        d = self.art / "decoder_lm" / self._lm_hash()
        if self._cached(d, "decoder_lm"):
            log(f"[decoder_lm] cached {d}")
        else:
            log(f"[decoder_lm] training -> {d}")
            lm_cfg = {"seed": self.cfg.get("seed", 0), **self.cfg.get("decoder_lm", {})}
            train_word_lm(self.words["train"], self.words["val"], lm_cfg, d)
        return d

    def compressor(self):
        if self._comp is None:
            probe = build_compressor(self._comp_cfg(), None)
            lm_dir = self.stage_decoder_lm() if probe.needs_lm else None
            self._comp = build_compressor(self._comp_cfg(), lm_dir)
            if self._comp.needs_vocab:
                self._comp.fit(self.words["train"])
        return self._comp

    # ---------------- stage: compressed corpus ----------------
    def corpus_dir(self):
        comp_probe = build_compressor(self._comp_cfg(), None)
        parts = [self.cfg["data"], self.cfg["compressor"]]
        if comp_probe.needs_lm:
            parts.append(self._lm_hash())
        return self.art / "corpus" / stage_hash(*parts)

    def stage_corpus(self):
        d = self.corpus_dir()
        if self._cached(d, "corpus"):
            log(f"[corpus] cached {d}")
            return d
        comp = self.compressor()
        workers = self.cfg["compressor"].get("workers", 0)
        stats = {"compressor": comp.describe()}
        flops_words = 0
        for split in ("train", "val", "test"):
            words = self.words[split]
            log(f"[corpus] compressing {split}: {len(words):,} words")
            if split == "test":
                rep, toks = compression_report(comp, words, workers=workers, progress=True)
                stats["test"] = rep
                secs = rep["compress_seconds"]
            else:
                t0 = time.time()
                toks = comp.compress(words, workers=workers, progress=True)
                secs = time.time() - t0
                oc, cc = n_chars(words), n_chars(toks)
                stats[split] = {"n_words": len(words), "orig_chars": oc, "comp_chars": cc, "char_ratio": cc / oc,
                                "escape_rate": sum(t.startswith("#") for t in toks) / len(toks),
                                "compress_seconds": secs}
                if split == "train":                       # spot-check losslessness on a training prefix
                    m = min(len(words), 64 * 256)
                    back = comp.decompress(toks[:m])
                    stats[split]["prefix_word_accuracy"] = float(np.mean([a == b for a, b in zip(back, words[:m])]))
            flops_words += len(words)
            _write_tokens(d / f"{split}.txt", toks)
        stats["compress_flops"] = comp.flops_per_word() * flops_words
        _write_json(d / "stats.json", stats)
        (d / "DONE").write_text("ok")
        t = stats["test"]
        log(f"[corpus] test: chars/word {t['chars_per_word_orig']:.2f} -> {t['chars_per_word_comp']:.2f}, "
            f"escapes {t['escape_rate']:.1%}, word accuracy {t['word_accuracy']:.4f}, lossless {t['lossless']}")
        return d

    # ---------------- stage: tokenizer ----------------
    def stage_tokenizer(self):
        cdir = self.stage_corpus()
        tcfg = self.cfg["tokenizer"]
        d = self.art / "tokenizer" / stage_hash(cdir.name, tcfg)
        if self._cached(d, "tokenizer"):
            log(f"[tokenizer] cached {d}")
            return d
        toks = {s: _read_tokens(cdir / f"{s}.txt") for s in ("train", "val", "test")}
        lines = to_lines(toks["train"], LINE_WORDS)
        if tcfg.get("max_train_lines"):
            lines = lines[: tcfg["max_train_lines"]]
        log(f"[tokenizer] training {tcfg['kind']} vocab {tcfg['vocab_size']} on {len(lines):,} lines")
        t0 = time.time()
        tok = train_tokenizer(lines, tcfg["kind"], tcfg["vocab_size"], tcfg.get("min_frequency", 2), tcfg.get("params"))
        secs = time.time() - t0
        tok.save(str(d / "tokenizer.json"))
        ids = {s: encode_words(tok, toks[s]) for s in toks}
        for s, a in ids.items():
            np.save(d / f"{s}_ids.npy", a)
        w = self.words
        rep = tokenizer_report(tok, ids["train"], ids["test"], toks["test"], n_chars(w["test"]), len(w["test"]))
        rep["train_seconds"] = secs
        rep["orig_chars_per_token"] = {s: n_chars(w[s]) / len(ids[s]) for s in ids}
        _write_json(d / "stats.json", rep)
        (d / "DONE").write_text("ok")
        log(f"[tokenizer] vocab {rep['vocab_size']}, {rep['tokens_per_orig_char']:.3f} tokens/char, "
            f"{rep['tokens_per_orig_word']:.2f} tokens/word, unk {rep['unk_rate']:.2%}")
        return d

    # ---------------- stage: LLM training + evaluation ----------------
    def stage_train(self):
        tdir = self.stage_tokenizer()
        cdir = self.corpus_dir()
        run = self.runs / self.name
        if (run / "metrics.json").exists() and not self._forced("train"):
            log(f"[train] cached {run}")
            return _read_json(run / "metrics.json")
        run.mkdir(parents=True, exist_ok=True)
        (run / "config.yaml").write_text(yaml.safe_dump(self.cfg, sort_keys=False))
        lcfg = self.cfg["llm"]
        device = get_device(lcfg.get("device", "auto"))
        ids = {s: np.load(tdir / f"{s}_ids.npy") for s in ("train", "val", "test")}
        tstats, cstats = _read_json(tdir / "stats.json"), _read_json(cdir / "stats.json")
        cpt = tstats["orig_chars_per_token"]
        V = tstats["vocab_size"]
        w = self.words
        set_seed(self.cfg.get("seed", 0))
        lcfg = {"seed": self.cfg.get("seed", 0), **lcfg}
        model, ts = train_llm(lcfg, ids["train"], ids["val"], V, cpt["train"], cpt["val"], run, device)

        nats, cnt = eval_nll(model, ids["test"], lcfg["ctx"], device, max_tokens=lcfg.get("test_tokens"),
                             precision=lcfg.get("precision", "auto"))
        test_bpc = bits_per_char(nats, cnt, cpt["test"])
        n_test_words = len(w["test"])
        evals = {"test_bpc": test_bpc, "test_nll_per_token": nats / cnt, "test_ppl_per_token": math.exp(nats / cnt),
                 "test_bits_per_orig_word": nats / math.log(2) / cnt * len(ids["test"]) / n_test_words,
                 "scored_tokens": cnt}

        comp = self.compressor()
        tok = load_tokenizer(tdir / "tokenizer.json")
        test_comp = _read_tokens(cdir / "test.txt")
        bench = inference_bench(model, comp, ids["test"], test_comp, n_chars(w["test"]), n_test_words,
                                lcfg["ctx"], device, **self.cfg.get("bench", {}))
        chunk = getattr(comp, "chunk", 256)
        samples = generate_samples(model, tok, comp, test_comp, w["test"], device, ctx=lcfg["ctx"], chunk=chunk)
        _write_json(run / "samples.json", samples)

        lm_stats = _read_json(self.art / "decoder_lm" / self._lm_hash() / "stats.json") if comp.needs_lm else None
        flops = {
            "decoder_lm_train": lm_stats["train_flops"] if lm_stats else 0,
            "corpus_compress": cstats["compress_flops"],
            "llm_train": ts["train_flops"],
        }
        flops["total_pipeline"] = sum(flops.values())
        flops["llm_train_per_orig_char"] = ts["train_flops"] / ts["orig_chars_seen"]
        metrics = {
            "name": self.name, "compressor": comp.describe(), "tokenizer_cfg": self.cfg["tokenizer"], "llm_cfg": lcfg,
            "hardware": hardware_info(), "device": str(device),
            "data": {k: len(v) for k, v in w.items()},
            "decoder_lm": lm_stats, "corpus": cstats, "tokenizer": tstats, "llm": ts,
            "eval": evals, "bench": bench, "flops": flops,
        }
        _write_json(run / "metrics.json", metrics)
        log(f"[train] {self.name}: test bpc {test_bpc:.4f}  tokens/char {tstats['tokens_per_orig_char']:.3f}  "
            f"train {ts['train_seconds']:.0f}s  decode {bench['decode_ms_per_token']:.2f} ms/token")
        return metrics

    def run(self):
        order = ["decoder_lm", "corpus", "tokenizer", "train"]
        until = self.until or "train"
        fn = {"decoder_lm": lambda: self.compressor(), "corpus": self.stage_corpus,
              "tokenizer": self.stage_tokenizer, "train": self.stage_train}
        return fn[until]() if until in order else None


def run_experiment(cfg, until=None, force=False):
    return Experiment(cfg, force=force, until=until).run()


def tokenizer_sweep(cfg, kinds, vocab_sizes, with_raw=True, max_lines=100_000):
    """Train a grid of tokenizers on the compressed corpus (and the raw text) and tabulate quality."""
    exp = Experiment(cfg)
    cdir = exp.stage_corpus()
    w = exp.words
    oc, ow = n_chars(w["test"]), len(w["test"])
    rows = []
    sources = [("compressed:" + cfg["compressor"]["kind"], _read_tokens(cdir / "train.txt"), _read_tokens(cdir / "test.txt"))]
    if with_raw:
        sources.append(("raw", w["train"], w["test"]))
    for name, tr, te in sources:
        log(f"[sweep] source {name}")
        lines = to_lines(tr, LINE_WORDS)[:max_lines]
        used = tr[: len(lines) * LINE_WORDS]
        rows += sweep(lines, used, te, oc, ow, kinds, vocab_sizes, name)
    out = exp.runs / f"{exp.name}_tokenizer_sweep"
    exp.runs.mkdir(parents=True, exist_ok=True)
    _write_json(str(out) + ".json", rows)
    cols = ["source", "kind", "vocab_size", "tokens_per_orig_char", "tokens_per_orig_word", "compressed_chars_per_token",
            "unk_rate", "roundtrip_lines_ok", "unigram_bits_per_char", "bigram_bits_per_char", "train_seconds"]
    lines_md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines_md.append("| " + " | ".join(f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    Path(str(out) + ".md").write_text("\n".join(lines_md) + "\n")
    print("\n".join(lines_md))
    return rows


def phrase_sweep(cfg, words, phrases, max_ns, bpe_ref=(), min_count=20, fallback=1000, max_lines=100_000):
    """Grid over (word tokens x phrase tokens x max phrase length) on the RAW text, next to BPE references.
    Reports tokens per original word / character, where the tokens come from, and the bigram proxy."""
    exp = Experiment(cfg)
    w = exp.words
    oc, ow = n_chars(w["test"]), len(w["test"])
    lines = to_lines(w["train"], LINE_WORDS)[:max_lines]
    used = w["train"][: len(lines) * LINE_WORDS]
    specs = [("bpe", {"vocab_size": v}) for v in bpe_ref]
    specs += [("phrase", {"words": a, "phrases": b, "max_n": n})
              for a in words for b in phrases for n in (max_ns if b > 0 else max_ns[:1])]
    rows = []
    for kind, p in specs:
        t0 = time.time()
        if kind == "bpe":
            tok = train_tokenizer(lines, "bpe", p["vocab_size"])
            label = f"bpe-{p['vocab_size']}"
        else:
            tok = PhraseTokenizer.fit(lines, fallback=fallback, min_count=min_count, **p)
            label = f"words{p['words']}+phrases{p['phrases']}(n<={p['max_n']})"
        secs = time.time() - t0
        tr_ids, te_ids = encode_words(tok, used), encode_words(tok, w["test"])
        rep = tokenizer_report(tok, tr_ids, te_ids, w["test"], oc, ow)
        rows.append({"tokenizer": label, "fit_seconds": secs, "epoch_tokens_M": len(tr_ids) / 1e6, **rep})
        log(f"[phrase-sweep] {label}: vocab {rep['vocab_size']}, {rep['tokens_per_orig_word']:.3f} tokens/word")
    out = exp.runs / f"{exp.name}_phrase_sweep"
    exp.runs.mkdir(parents=True, exist_ok=True)
    _write_json(str(out) + ".json", rows)
    cols = ["tokenizer", "vocab_size", "tokens_per_orig_word", "tokens_per_orig_char", "token_frac_phrase",
            "token_frac_fallback_init", "unk_rate", "roundtrip_lines_ok", "bigram_bits_per_char", "epoch_tokens_M",
            "fit_seconds"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        md.append("| " + " | ".join(f"{r.get(c, '-'):.4g}" if isinstance(r.get(c), float) else str(r.get(c, "-"))
                                    for c in cols) + " |")
    Path(str(out) + ".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return rows
