"""YAML experiment configs with `extends:` inheritance and dotted command-line overrides."""
import copy
import hashlib
import json
from pathlib import Path

import yaml


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = deep_merge(_load(path.parent / parent), cfg)
    return cfg


def set_dotted(cfg, dotted, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path, overrides=()):
    """Load a config; `overrides` are strings like 'llm.batch_size=32' (values parsed as YAML)."""
    cfg = _load(path)
    cfg.setdefault("name", Path(path).stem)
    for ov in overrides:
        key, _, val = ov.partition("=")
        set_dotted(cfg, key.strip(), yaml.safe_load(val))
    return cfg


def stage_hash(*objs):
    """Stable short hash used to key cached artifacts by the config sections they depend on."""
    return hashlib.sha1(json.dumps(objs, sort_keys=True, default=str).encode()).hexdigest()[:10]
