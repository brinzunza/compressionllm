# compressionllm

Does training a language model on **learned, context-aware letter-dropping compression** make it
cheaper without making it worse?

The idea: replace each word by a short subsequence of its letters (`unique` → `uiqu`), and let a
language model put the letters back using the surrounding words. The encoder can drop more letters
where the context makes the word predictable (`the` → `t`) and fewer where it doesn't. If the
compressed text has fewer tokens per character, a model trained directly on it should need fewer
FLOPs per unit of text, at the cost of running a compressor and decompressor.

This repo is a modular pipeline for testing that claim end to end, with matched budgets:

```
text8 ─► compressor ─► tokenizer ─► LLM training ─► decompressor ─► metrics
          (raw | fixed rule | learned)   (BPE/unigram/word)   (GPT-2 style)   (restores the text)
```

## Findings so far

Earlier prototype (kept in git history at `5fa8ba2`; 300k held-out words, bigram decoder, then a
6.8M-parameter LSTM decoder):

| decoder | raw characters | xz size |
|---|---|---|
| bigram | −38% | −9% |
| LSTM | −40% | −13% |

Characters saved overstate storage saved, because the abbreviated text is denser and a generic
compressor finds less to squeeze. Whether the saving survives tokenization and LLM training is what this
pipeline measures; no results for that yet. Treat every number from a run as one seed at small scale.

## Quick start (GPU VM)

```bash
git clone https://github.com/brinzunza/compressionllm.git && cd compressionllm
scripts/setup_vm.sh                 # venv, dependencies, text8 download + checksum, unit tests
source .venv/bin/activate
make smoke                          # ~minutes: 3 tiny end-to-end runs + report (checks the whole pipeline)
make experiments                    # the four real experiments, then runs/REPORT.md
```

Everything is also available as plain commands:

```bash
cllm run configs/exp_learned.yaml                  # full pipeline for one config (stages are cached)
cllm run configs/exp_raw.yaml configs/exp_learned.yaml
cllm compress configs/exp_learned.yaml             # stop after the compressed corpus
cllm tokenize configs/exp_learned.yaml             # ...after the tokenizer
cllm tokenizer-sweep configs/exp_learned.yaml      # grid of tokenizers, compressed vs raw
cllm encode configs/exp_learned.yaml in.txt out.txt   # compress your own text (lowercase words)
cllm decode configs/exp_learned.yaml out.txt back.txt
cllm report                                        # runs/REPORT.md + runs/report.csv
cllm run configs/exp_learned.yaml -s llm.budget.value=300000000 -s llm.n_layer=8   # override anything
```

Requires Python ≥ 3.10 and a PyTorch build for your GPU (on Linux the default `pip` wheel is the CUDA
build). The CPU/MPS path works too, only slowly. The GPU code paths (bf16, fused AdamW) are exercised for
the first time on a CUDA machine, so run `make smoke` first.

## The three stages you iterate on

**1. Compressor** (`src/cllm/compressors/`). Maps words to one token per word and back. Registered kinds:

| kind | what it does | lossless |
|---|---|---|
| `identity` | raw text (baseline) | yes |
| `rule` (`rule: vowel_drop` or `trunc`, `k`) | context-free letter removal, decoded by the LM; `verify: true` escapes words the decoder would get wrong | only with `verify: true` |
| `learned` | shortest letter subsequence the decoder LM still resolves, with logit margin `tau`; context decides how much to drop | yes (escapes `#word` otherwise) |

To try your own, subclass `Compressor` and `@register("name")` it (see `compressors/base.py`); set
`needs_lm = True` to reuse the pipeline's word-level LSTM decoder prior.

**2. Training** (`src/cllm/llm.py`). A GPT-2 style transformer (Hugging Face `GPT2LMHeadModel`) trained
on the token stream. The same config trains the raw and compressed variants. `llm.budget` decides what
"comparable compute" means: `tokens` (≈ equal FLOPs, the default), `orig_chars` (equal text seen), or
`flops`.

**3. Decompressor** (same compressor classes). Every evaluation decodes the compressed test stream back
to text and checks it, and `generate_samples` decompresses what the LLM generates
(`runs/<name>/samples.json`). Out-of-vocabulary or invalid abbreviations in generated text are kept literal.

## Tokenizer iteration

Tokenizers (`bpe`, `unigram`, `wordlevel`; Hugging Face `tokenizers`) all split on spaces, so word
boundaries survive. `cllm tokenizer-sweep CONFIG --kinds bpe,unigram --vocab 1000,4000,16000` trains a grid
on the compressed corpus and on raw text and prints, per tokenizer: tokens per original character and
word, compressed characters per token, `<unk>` rate, exact round-trip rate, and two cheap proxies for how
modelable the tokens are (unigram and bigram cross-entropy in bits per **original** character). Rows with
`unk_rate > 0` or `roundtrip_lines_ok < 1` are lossy and not comparable. Add a new tokenizer kind in
`src/cllm/tokenizer.py:train_tokenizer`.

## Metrics

Written to `runs/<name>/metrics.json`, tabulated by `cllm report` (with percentage change vs the raw run):

- **Compression**: characters per word, escape rate, token reduction (tokens per original character), classic
  zlib/bz2/xz sizes of original vs compressed text.
- **Reconstruction**: lossless flag, word accuracy, character accuracy; by word length.
- **Language modelling**: test bits per original character (the headline), perplexity per token, bits per word,
  and a validation curve against steps, tokens, original characters, FLOPs and wall time (`llm.curve`).
- **Cost**: analytic FLOPs for decoder-LM training, corpus compression and LLM training (summed into
  `total_pipeline`), parameter counts, wall time and tokens/s, peak memory for training and inference.
- **Inference**: prefill latency, decode ms/token with KV cache, and the full per-1k-original-characters cost
  including compress and decompress time.

Notes on reading them: bits per original character is only a valid cross-pipeline number for lossless
runs. The decoder LM's own size and training cost are reported but not folded into the bpc. FLOPs are analytic
estimates (Kaplan-style, see `flops.py`), not profiler counts.

## Reproducibility and caveats

- The encoder simulates the decoder, so encoder and decoder must produce the same logits. Both run in fp32
  with TF32 disabled and `tau` (default 0.1) adds a safety margin, but decoding on different hardware or
  PyTorch versions is not guaranteed. Every run verifies a full decode of the test stream.
- Context resets every 256 words (`decoder_lm.chunk`), which lets chunks be encoded and decoded in parallel.
- Encoding is a Python search; `compressor.workers: N` runs it in N CPU processes (Linux).
- Single seed, small models, one corpus (text8: lowercase letters and spaces). Differences of a few percent are
  within noise until repeated with several seeds.

## Layout

```
configs/      base.yaml, exp_*.yaml (raw / fixed / learned), smoke*.yaml (tiny)
src/cllm/     data, word_lm, compressors/, tokenizer, llm, evaluate, bench, pipeline, report, cli
tests/        unit tests (pytest): rules, codec losslessness, tokenizers, eval bookkeeping
scripts/      setup_vm.sh, get_data.sh
results/      training log of the earlier prototype
artifacts/    (generated) cached decoder LMs, compressed corpora, tokenizers
runs/         (generated) per-experiment metrics, samples, model weights
```
