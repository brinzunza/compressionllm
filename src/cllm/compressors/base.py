"""Compressor interface.

A compressor maps a list of words to a list of tokens, one token per word, and back. A token is
either a (shortened) word or an escape `#word` that stores the word in full. Everything downstream
(tokenizer, LLM training, metrics) only sees the space-joined token stream, so any compressor that
implements this interface plugs into the pipeline.

To add one: subclass Compressor, implement compress/decompress, and register it:

    from cllm.compressors import register
    @register("my_compressor")
    class MyCompressor(Compressor): ...

then use `compressor: {kind: my_compressor, ...}` in a config. Set `needs_lm = True` if it uses the
word-level decoder LM (the pipeline then trains one and sets `self.lm_dir`).
"""


class Compressor:
    needs_lm = False
    lm_dir = None

    def compress(self, words, workers=0, progress=False):
        raise NotImplementedError

    def decompress(self, tokens, progress=False):
        raise NotImplementedError

    def flops_per_word(self):
        """Forward FLOPs the decompressor spends per word (0 if it is a lookup/rule)."""
        return 0

    def describe(self):
        return {"kind": type(self).__name__}


class Identity(Compressor):
    """No compression: the raw-text baseline."""

    def __init__(self, **_ignored):    # tolerate params inherited from an `extends:` parent config
        pass

    def compress(self, words, workers=0, progress=False):
        return list(words)

    def decompress(self, tokens, progress=False):
        return list(tokens)

    def describe(self):
        return {"kind": "identity"}
