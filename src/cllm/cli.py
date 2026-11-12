"""Command line: `cllm <command> configs/exp.yaml [-s key=value ...]`."""
import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .utils import hardware_info


def _cfg(args):
    return load_config(args.config, args.set or [])


def _add_cfg(p, force=True):
    p.add_argument("config", help="experiment YAML (may `extends:` another)")
    p.add_argument("-s", "--set", action="append", metavar="KEY=VALUE",
                   help="override a config value, e.g. -s llm.budget.value=50000000 (repeatable)")
    if force:
        p.add_argument("--force", action="store_true", help="redo the requested (final) stage even if cached")


def cmd_run(args, until):
    from .pipeline import run_experiment
    for path in args.config:
        cfg = load_config(path, args.set or [])
        run_experiment(cfg, until=until or args.until, force=args.force)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cllm", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run the pipeline for one or more configs (stages are cached)")
    p.add_argument("config", nargs="+")
    p.add_argument("-s", "--set", action="append", metavar="KEY=VALUE")
    p.add_argument("--until", choices=["decoder_lm", "corpus", "tokenizer", "train"], default="train",
                   help="stop after this stage (default: train = everything)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=lambda a: cmd_run(a, None))

    for name, until, help_ in [("compress", "corpus", "train the decoder LM (if needed) and compress the corpus"),
                               ("tokenize", "tokenizer", "...and train the tokenizer and encode the corpus"),
                               ("train", "train", "...and train + evaluate + benchmark the LLM")]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("config", nargs="+")
        p.add_argument("-s", "--set", action="append", metavar="KEY=VALUE")
        p.add_argument("--force", action="store_true")
        p.set_defaults(fn=lambda a, u=until: cmd_run(a, u))

    p = sub.add_parser("tokenizer-sweep", help="grid over tokenizer kind x vocab size on the compressed (and raw) text")
    _add_cfg(p, force=False)
    p.add_argument("--kinds", default="bpe,unigram,wordlevel")
    p.add_argument("--vocab", default="500,1000,2000,4000,8000,16000")
    p.add_argument("--no-raw", action="store_true", help="skip the raw-text reference tokenizers")
    p.add_argument("--max-lines", type=int, default=100_000, help="training lines per tokenizer (64 words each)")

    def sweep_fn(a):
        from .pipeline import tokenizer_sweep
        tokenizer_sweep(_cfg(a), a.kinds.split(","), [int(v) for v in a.vocab.split(",")],
                        with_raw=not a.no_raw, max_lines=a.max_lines)
    p.set_defaults(fn=sweep_fn)

    p = sub.add_parser("phrase-sweep", help="grid over word tokens x phrase tokens x max phrase length (raw text)")
    _add_cfg(p, force=False)
    p.add_argument("--words", default="16000", help="comma list: number of whole-word tokens")
    p.add_argument("--phrases", default="0,4000,16000,32000", help="comma list: number of multi-word tokens (0 = word-level)")
    p.add_argument("--max-n", default="3,4", help="comma list: longest phrase in words")
    p.add_argument("--bpe-ref", default="8000,32000", help="comma list of BPE vocab sizes to compare against ('' = none)")
    p.add_argument("--min-count", type=int, default=20)
    p.add_argument("--max-lines", type=int, default=100_000, help="training lines (64 words each)")

    def phrase_fn(a):
        from .pipeline import phrase_sweep
        ints = lambda v: [int(x) for x in v.split(",") if x]
        phrase_sweep(_cfg(a), ints(a.words), ints(a.phrases), ints(a.max_n), ints(a.bpe_ref),
                     min_count=a.min_count, max_lines=a.max_lines)
    p.set_defaults(fn=phrase_fn)

    for name, help_ in [("encode", "compress a text file (space-separated lowercase words)"),
                        ("decode", "decompress a file produced by `encode`")]:
        p = sub.add_parser(name, help=help_)
        _add_cfg(p, force=False)
        p.add_argument("input")
        p.add_argument("output", nargs="?", help="default: stdout")
        p.set_defaults(fn=lambda a, n=name: cmd_codec(a, n))

    p = sub.add_parser("report", help="aggregate runs/*/metrics.json into REPORT.md / report.csv")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--baseline", help="run name to compare against (default: the identity-compressor run)")
    p.set_defaults(fn=lambda a: __import__("cllm.report", fromlist=["x"]).write_report(a.runs_dir, a.baseline))

    p = sub.add_parser("hw", help="print hardware / torch info")
    p.set_defaults(fn=lambda a: print(json.dumps(hardware_info(), indent=2)))

    args = ap.parse_args(argv)
    args.fn(args)


def cmd_codec(args, name):
    from .pipeline import Experiment
    exp = Experiment(_cfg(args))
    comp = exp.compressor()
    text = Path(args.input).read_text().split()
    out = comp.compress(text, progress=True) if name == "encode" else comp.decompress(text, progress=True)
    s = " ".join(out) + "\n"
    if args.output:
        Path(args.output).write_text(s)
    else:
        sys.stdout.write(s)


if __name__ == "__main__":
    main()
