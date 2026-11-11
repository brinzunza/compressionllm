"""Compressor registry. `build_compressor({'kind': ..., **params}, lm_dir)` is what the pipeline calls."""
from .base import Compressor, Identity
from .lm_codec import LMCodec, RULES

REGISTRY = {}


def register(name):
    def deco(cls):
        REGISTRY[name] = cls
        return cls
    return deco


register("identity")(Identity)


@register("learned")
def _learned(**kw):
    return LMCodec(scheme="subseq", **kw)


@register("rule")
def _rule(**kw):
    return LMCodec(scheme="rule", **kw)


def build_compressor(cfg, lm_dir=None):
    cfg = dict(cfg)
    kind = cfg.pop("kind")
    if kind not in REGISTRY:
        raise ValueError(f"unknown compressor '{kind}'; registered: {sorted(REGISTRY)}")
    comp = REGISTRY[kind](**cfg)
    if comp.needs_lm:
        comp.lm_dir = lm_dir
    return comp


__all__ = ["Compressor", "Identity", "LMCodec", "RULES", "register", "build_compressor", "REGISTRY"]
