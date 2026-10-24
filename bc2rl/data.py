"""`bc2rl generate`: build frozen splits and their manifest under $PROJ_DATA/data/<version>/."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from collections import Counter

from . import task
from .runtime import atomic_json


def data_dir(cfg):
    return os.path.join(cfg["proj_data"], "data", cfg["data"]["version"])


def load_data(cfg, splits):
    d = data_dir(cfg)
    man = json.load(open(os.path.join(d, "manifest.json")))
    out = {}
    for s in splits:
        probs = task.load_split(os.path.join(d, f"{s}.jsonl"))
        h = task.manifest_hash(probs)
        if h != man["splits"][s]["sha256"]:
            raise SystemExit(f"split {s} hash {h} != manifest {man['splits'][s]['sha256']}")
        out[s] = probs
    out["format"] = task.load_split(os.path.join(d, "format.jsonl"))
    return out, man


def describe(problems):
    ops = Counter(o for p in problems for o in p.ops_used)
    tbins = Counter(min(p.target // 100, 9) for p in problems)
    return {
        "count": len(problems),
        "by_inputs": dict(Counter(p.n_inputs for p in problems)),
        "structure": dict(Counter(f"{p.n_inputs}:{p.structure}" for p in problems)),
        "op_freq": dict(ops),
        "target_bins_100": {str(k * 100): v for k, v in sorted(tbins.items())},
        "sha256": task.manifest_hash(problems),
    }


def source_sha(module):
    return hashlib.sha256(inspect.getsource(module).encode()).hexdigest()


def cmd_generate(cfg):
    d = data_dir(cfg)
    if os.path.exists(os.path.join(d, "manifest.json")):
        print(f"data {d} exists; refusing to overwrite (frozen). Verifying hashes.")
        load_data(cfg, ["dev", "devdemo", "bc", "online", "val", "test"])
        print("hashes OK")
        return 0
    g = cfg["data"]
    gcfg = task.GenConfig(min_value=g["min_value"], max_value=g["max_value"], max_target=g["max_target"],
                          max_abs_intermediate=g["max_abs_intermediate"], ops=tuple(g["ops"]))
    splits, stats = task.generate_splits(g["main_seed"], g["dev_seed"], gcfg, tuple(g["horizons"]),
                                         g.get("sizes"), g.get("format_example_inputs", 4))
    os.makedirs(d, exist_ok=True)
    for s, probs in splits.items():
        task.save_split(os.path.join(d, f"{s}.jsonl"), probs)
    keys = {}
    for s, probs in splits.items():
        for p in probs:
            if p.key in keys:
                raise SystemExit(f"duplicate canonical key across {keys[p.key]} and {s}")
            keys[p.key] = s
    man = {
        "version": g["version"] if "version" in g else cfg["data"]["version"],
        "generator": task.GENERATOR_VERSION, "verifier": task.VERIFIER_VERSION,
        "task_source_sha256": source_sha(task),
        "gen_config": g, "rejection_stats": stats,
        "splits": {s: describe(p) for s, p in splits.items()},
        "format_example": splits["format"][0].to_json(),
    }
    atomic_json(os.path.join(d, "manifest.json"), man)
    try:
        man["token_stats"] = token_stats(cfg, splits)
        atomic_json(os.path.join(d, "token_stats.json"), man["token_stats"])
    except (OSError, KeyError) as e:          # tokenizer not available (unit tests)
        print(f"token stats skipped: {e}")
    print(json.dumps({s: v["by_inputs"] for s, v in man["splits"].items()}))
    out = cfg.get("_results_dir")
    if out:
        atomic_json(os.path.join(out, "data_manifest.json"), man)
    return 0


def token_stats(cfg, splits):
    """Reference-completion and prompt token lengths by split and input count (cap check)."""
    from .modeling import chat_prompt, load_tokenizer, model_info
    tok = load_tokenizer(model_info(cfg["proj_data"])["path"])
    fmt = splits["format"][0]
    out = {}
    for s in ("devdemo", "bc", "val", "dev"):
        for h in sorted({p.n_inputs for p in splits[s]}):
            ps = [p for p in splits[s] if p.n_inputs == h]
            comp = [len(tok(task.completion_text(p), add_special_tokens=False).input_ids) + 1 for p in ps]
            prm = [len(tok(chat_prompt(tok, task.prompt_text(p, "format", fmt)), add_special_tokens=False).input_ids)
                   for p in ps]
            out[f"{s}:{h}"] = {"ref_completion_max": max(comp), "ref_completion_mean": sum(comp) / len(comp),
                               "prompt_format_max": max(prm)}
    return out
