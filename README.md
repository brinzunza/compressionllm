# compressionllm

Lossless text compression by **dropping letters and letting a language model put them back**.

Each word is replaced by a short subsequence of its letters (first letter always kept), for example
`unique` → `uiqu`. A decoder looks at the previously decoded words and picks the most likely
vocabulary word consistent with the abbreviation. The encoder runs a copy of the decoder, so it
knows exactly which letters are safe to drop: for every word it keeps the shortest abbreviation
that still decodes to the true word. If none works, the word is stored in full with a `#` prefix.

Because the decoder is deterministic and sees only already-decoded words, the round trip is exact.

## Results

Corpus: [text8](http://mattmahoney.net/dc/textdata.html) (first 90M characters for training,
held-out text after that for testing). Vocabulary: 20,000 most frequent words; other words are escaped.

| decoder | test words | raw chars | zlib -9 | xz -9 |
|---|---|---|---|---|
| bigram + unigram (`abbrev.py`) | 300,000 | −38% | −18% | −9% |
| 6.8M-param LSTM (`neural_abbrev.py`) | 51,200 | −40% | −18% | −13% |

Neural run: 0 round-trip mismatches, 14.2% of words escaped (6.6% are out of vocabulary),
mean 3.55 characters per word vs 5.94 originally. Sizes are for the abbreviated text compared with the
original text under the same compressor; the model itself (27 MB) is not counted.

**The main finding is a negative one.** Characters saved overstate the storage saved: abbreviated text
is denser, so a general-purpose compressor finds less to squeeze. And the same LSTM used as an
arithmetic coder would do better than letter-dropping: its ideal code length on the test text is
about 64.7 KB against 78.7 KB for abbreviation + xz and 90.8 KB for xz alone (about −29% vs xz).
That figure is computed from the model's probabilities, not produced by a real arithmetic coder, and it
spells out-of-vocabulary words at a flat log2(27) bits per character. Exact numbers will vary with
the training run.

## How it works

- `abbrev.py` – baseline. Interpolated bigram/unigram decoder; the encoder searches subsequences
  shortest-first. Run: `python3 abbrev.py 300000`.
- `lm.py`, `train_lm.py` – word-level LSTM language model (embedding 256, hidden 512, tied output
  embedding). Context is reset every 256 words so chunks can be encoded and decoded in parallel.
  Training uses a wall-clock budget; the log of the reported run is in `results/train.log`
  (validation perplexity 164).
- `neural_abbrev.py` – encoder and decoder. The decoder masks the LM's logits to words containing the
  abbreviation as a subsequence and takes the argmax. The encoder accepts an abbreviation only if the
  true word wins by a logit margin `TAU` (default 0.05), so small floating-point differences between
  batched encoding and step-by-step decoding cannot change the result.

## Reproduce

```bash
pip install -r requirements.txt
scripts/get_data.sh                 # downloads and checksums text8 into data/
python3 abbrev.py 300000            # bigram baseline
mkdir -p models
python3 train_lm.py 40              # trains for 40 minutes (uses MPS if available), writes models/
python3 neural_abbrev.py 200 0.05   # 200 chunks of 256 test words, margin 0.05
```

## Ideas not tried yet

- A trained encoder that imitates the search (the current encoder searches at encode time).
- Choosing abbreviations to minimise bits under the model rather than characters.
- A real arithmetic coder driven by the same LM, as the proper baseline.
- A character-aware decoder to reduce the 6.6% of out-of-vocabulary words.
