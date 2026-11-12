"""Aggregate runs/*/metrics.json into one comparison table (markdown + CSV)."""
import csv
import json
from pathlib import Path

# (column header, getter, format)
COLUMNS = [
    ("run", lambda m: m["name"], "{}"),
    ("compressor", lambda m: _comp_label(m["compressor"]), "{}"),
    ("tok", lambda m: f"{m['tokenizer_cfg']['kind']}-{m['tokenizer']['vocab_size']}", "{}"),
    ("chars/word", lambda m: m["corpus"]["test"]["chars_per_word_comp"], "{:.2f}"),
    ("escape%", lambda m: 100 * m["corpus"]["test"]["escape_rate"], "{:.1f}"),
    ("lossless", lambda m: "yes" if m["corpus"]["test"]["lossless"] else "NO", "{}"),
    ("recon acc", lambda m: m["corpus"]["test"]["word_accuracy"], "{:.4f}"),
    ("tok/char", lambda m: m["tokenizer"]["tokens_per_orig_char"], "{:.3f}"),
    ("test bpc", lambda m: m["eval"]["test_bpc"], "{:.4f}"),
    ("ppl/token", lambda m: m["eval"]["test_ppl_per_token"], "{:.1f}"),
    ("LLM params(M)", lambda m: m["llm"]["params_total"] / 1e6, "{:.2f}"),
    ("train tokens(M)", lambda m: m["llm"]["tokens_seen"] / 1e6, "{:.1f}"),
    ("orig chars seen(M)", lambda m: m["llm"]["orig_chars_seen"] / 1e6, "{:.1f}"),
    ("LLM train PFLOPs", lambda m: m["flops"]["llm_train"] / 1e15, "{:.3f}"),
    ("pipeline PFLOPs", lambda m: m["flops"]["total_pipeline"] / 1e15, "{:.3f}"),
    ("train s", lambda m: m["llm"]["train_seconds"], "{:.0f}"),
    ("train mem MB", lambda m: m["llm"]["peak_train_mem_mb"], "{:.0f}"),
    ("decode ms/tok", lambda m: m["bench"]["decode_ms_per_token"], "{:.2f}"),
    ("gen ms/1k chars", lambda m: m["bench"]["generate_ms_per_1k_orig_chars"], "{:.1f}"),
    ("decompress ms/1k chars", lambda m: m["bench"]["decompress_ms_per_1k_orig_chars"], "{:.1f}"),
    ("end-to-end ms/1k chars", lambda m: m["bench"]["end_to_end_ms_per_1k_orig_chars"], "{:.1f}"),
    ("infer mem MB", lambda m: m["bench"]["peak_infer_mem_mb"], "{:.0f}"),
]


def _comp_label(c):
    k = c.get("kind", "?")
    if k == "rule":
        return f"rule:{c.get('rule')}{'+verify' if c.get('verify') else ''}"
    return k


def load_runs(runs_dir="runs"):
    out = []
    for p in sorted(Path(runs_dir).glob("*/metrics.json")):
        out.append(json.loads(p.read_text()))
    return out


def build_table(runs):
    rows = []
    for m in runs:
        row = []
        for _, get, fmt in COLUMNS:
            try:
                row.append(fmt.format(get(m)))
            except (KeyError, TypeError):
                row.append("-")
        rows.append(row)
    return [c[0] for c in COLUMNS], rows


def relative_to_baseline(runs, baseline=None):
    """Percent change of key metrics vs the identity (raw) run, if there is one."""
    if baseline:
        base = next((m for m in runs if m["name"] == baseline), None)
    else:    # the raw run that saw the most tokens, so short test runs are never the baseline
        raw = [m for m in runs if m["compressor"].get("kind") == "identity"]
        base = max(raw, key=lambda m: m["llm"]["tokens_seen"], default=None)
    if base is None:
        return []
    keys = [("test bpc", lambda m: m["eval"]["test_bpc"]),
            ("tok/char", lambda m: m["tokenizer"]["tokens_per_orig_char"]),
            ("LLM train FLOPs", lambda m: m["flops"]["llm_train"]),
            ("pipeline FLOPs", lambda m: m["flops"]["total_pipeline"]),
            ("train s", lambda m: m["llm"]["train_seconds"]),
            ("end-to-end ms/1k chars", lambda m: m["bench"]["end_to_end_ms_per_1k_orig_chars"])]
    lines = [f"\nChange vs `{base['name']}` (negative = better than baseline):\n",
             "| run | " + " | ".join(k for k, _ in keys) + " |", "|---|" + "---|" * len(keys)]
    for m in runs:
        if m is base:
            continue
        cells = [f"{100 * (g(m) / g(base) - 1):+.1f}%" for _, g in keys]
        lines.append(f"| {m['name']} | " + " | ".join(cells) + " |")
    return lines


def write_report(runs_dir="runs", baseline=None):
    runs = load_runs(runs_dir)
    if not runs:
        print(f"no runs found in {runs_dir}/")
        return
    header, rows = build_table(runs)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md += relative_to_baseline(runs, baseline)
    if any(not m["corpus"]["test"]["lossless"] for m in runs):
        md.append("\nRuns marked lossless=NO decode to different text than the input, so their `test bpc` "
                  "(bits per original character) is not a valid compression figure; compare them on `recon acc` too.")
    text = "\n".join(md) + "\n"
    Path(runs_dir, "REPORT.md").write_text(text)
    with open(Path(runs_dir, "report.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(text)
