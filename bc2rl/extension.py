"""Amendment A5 extension: fresh confirmation panels (`bc2rl xdata`) and their evaluation (`bc2rl xeval`).

Protocol: protocol/AMENDMENT-A5-2026-10-26.md. Frozen v1 code paths are not changed: this module only reuses them.

`xdata` draws fresh problems with the frozen v1 generator configuration and a new seed, deduplicated by canonical
key against every v1 split, the format example and the mechanism splits. It writes `data/v1/ext/<name>/`.

`xeval` samples `attempts` independent completions per problem for each checkpoint and panel, keeps the full records
on PROJ_DATA and writes a compact per-problem count table (`xcounts.csv`) to the results directory:
c = all attempts; c_a = attempts [0, split); c_b = attempts [split, n) when `split` is set (initial adapters).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict

import numpy as np
import torch

from . import task
from .data import data_dir
from .evaluate import run_attempts, stable_int, summarize
from .mech import MECH_SPLITS, mech_dir
from .modeling import DECODING_RECORD, load_adapter, load_policy, load_tokenizer, model_info
from .runtime import CostLedger, atomic_json, write_jsonl_gz


def ext_dir(cfg, name):
    return os.path.join(data_dir(cfg), "ext", name)


def existing_keys(cfg):
    """Canonical keys of every v1 split, the format example and the mechanism splits."""
    d = data_dir(cfg)
    man = json.load(open(os.path.join(d, "manifest.json")))
    seen = set()
    for s in list(man["splits"]) + ["format"]:
        seen.update(p.key for p in task.load_split(os.path.join(d, f"{s}.jsonl")))
    for s in MECH_SPLITS:
        seen.update(p.key for p in task.load_split(os.path.join(mech_dir(cfg), f"{s}.jsonl")))
    return seen


def draw_fresh(seed, gcfg, panels, seen):
    """panels: {name: (n_inputs, count)}. Returns ({name: [Problem]}, stats). `seen` is extended in place."""
    rng = random.Random(seed)
    out, stats = {}, {}
    for name, (h, count) in panels.items():
        st = stats.setdefault(name, Counter())
        got = []
        while len(got) < count:
            values, target, trace, structure, ops_used = task.sample_problem(rng, h, gcfg, st)
            key = task.canonical_key(values, target)
            if key in seen:
                st["reject:duplicate"] += 1
                continue
            seen.add(key)
            shown = values[:]
            rng.shuffle(shown)
            got.append(task.Problem(pid=f"{name}-{h}-{len(got):05d}", split=name, n_inputs=h, inputs=shown,
                                    target=target, ref_trace=trace, structure=structure, ops_used=ops_used))
        out[name] = got
    return out, stats


def cmd_xdata(cfg):
    xd = cfg["xdata"]
    d = ext_dir(cfg, xd["name"])
    mpath = os.path.join(d, "manifest.json")
    if os.path.exists(mpath):
        load_ext(cfg, xd["name"])
        print(f"ext data {d} exists; hashes OK")
        return 0
    seen = existing_keys(cfg)
    n_existing = len(seen)
    g = cfg["data"]
    gcfg = task.GenConfig(min_value=g["min_value"], max_value=g["max_value"], max_target=g["max_target"],
                          max_abs_intermediate=g["max_abs_intermediate"], ops=tuple(g["ops"]))
    panels = {name: (int(v["n_inputs"]), int(v["count"])) for name, v in xd["panels"].items()}
    out, stats = draw_fresh(xd["seed"], gcfg, panels, seen)
    os.makedirs(d, exist_ok=True)
    for s, probs in out.items():
        task.save_split(os.path.join(d, f"{s}.jsonl"), probs)
    base = os.path.join(data_dir(cfg), "manifest.json")
    man = {"base_manifest_sha256": hashlib.sha256(open(base, "rb").read()).hexdigest(),
           "mech_manifest_sha256": hashlib.sha256(open(os.path.join(mech_dir(cfg), "mech_manifest.json"),
                                                       "rb").read()).hexdigest(),
           "generator": task.GENERATOR_VERSION, "gen_config": g, "seed": xd["seed"],
           "existing_keys_excluded": n_existing, "rejection_stats": {k: dict(v) for k, v in stats.items()},
           "panels": {s: {"count": len(p), "n_inputs": panels[s][0], "sha256": task.manifest_hash(p),
                          "structure": dict(Counter(x.structure for x in p)),
                          "op_freq": dict(Counter(o for x in p for o in x.ops_used))} for s, p in out.items()}}
    atomic_json(mpath, man)
    if cfg.get("_results_dir"):
        atomic_json(os.path.join(cfg["_results_dir"], "xdata_manifest.json"), man)
    print(json.dumps({s: {k: v[k] for k in ("count", "sha256")} for s, v in man["panels"].items()}))
    print(json.dumps(man["rejection_stats"]))
    return 0


def load_ext(cfg, name):
    d = ext_dir(cfg, name)
    man = json.load(open(os.path.join(d, "manifest.json")))
    out = {}
    for s, v in man["panels"].items():
        probs = task.load_split(os.path.join(d, f"{s}.jsonl"))
        if task.manifest_hash(probs) != v["sha256"]:
            raise SystemExit(f"ext panel {s} hash mismatch")
        out[s] = probs
    return out, man


def count_rows(recs, tag, panel, split):
    by = defaultdict(dict)
    meta = {}
    for r in recs:
        by[r["pid"]][int(r["attempt"])] = (int(r["reward"]), int(r["n_tokens"]), r["finish"] == "length")
        meta[r["pid"]] = int(r["n_inputs"])
    rows = []
    for pid in sorted(by):
        a = by[pid]
        n = len(a)
        if sorted(a) != list(range(n)):
            raise SystemExit(f"{tag}/{pid}: attempts not contiguous")
        rew = [a[i][0] for i in range(n)]
        row = {"tag": tag, "panel": panel, "pid": pid, "n_inputs": meta[pid], "n": n, "c": sum(rew),
               "c_a": sum(rew[:split]) if split else "", "c_b": sum(rew[split:]) if split else "",
               "mean_tokens": round(float(np.mean([a[i][1] for i in range(n)])), 1),
               "n_trunc": sum(a[i][2] for i in range(n))}
        rows.append(row)
    return rows


XCOLS = ["tag", "panel", "pid", "n_inputs", "n", "c", "c_a", "c_b", "mean_tokens", "n_trunc"]


def cmd_xeval(cfg):
    """Evaluate each checkpoint on each panel; skip (tag, panel) pairs whose summary matches adapter hash + protocol."""
    ev = cfg["xeval"]
    pd = cfg["proj_data"]
    data, man = load_ext(cfg, ev["data_name"])
    fmt = task.load_split(os.path.join(data_dir(cfg), "format.jsonl"))[0]
    info = model_info(pd)
    tok = load_tokenizer(info["path"])
    model = load_policy(info["path"], cfg["model"]["lora"], seed=0)
    out_dir = os.path.join(pd, "evals", "ext", ev["data_name"])
    os.makedirs(out_dir, exist_ok=True)
    ledger = CostLedger(os.path.join(out_dir, f"cost-{cfg['run_id']}.json"), "xeval")
    res_dir = cfg.get("_results_dir")
    for ck in ev["checkpoints"]:
        tag, adapter = ck["tag"], ck["adapter"]
        apath = adapter if os.path.isabs(adapter) else os.path.join(pd, adapter)
        asha = load_adapter(model, apath)
        if ck.get("sha256") and ck["sha256"] != asha:
            raise SystemExit(f"{tag}: adapter sha256 {asha} != registered {ck['sha256']}")
        for panel in ev["panels"]:
            attempts, split = int(ck["attempts"]), int(ck.get("split") or 0)
            proto = {"data": ev["data_name"], "panel": panel, "panel_sha256": man["panels"][panel]["sha256"],
                     "attempts": attempts, "split": split, "cap": ev["cap"], "variant": ev["variant"],
                     "decoding": DECODING_RECORD, "verifier": task.VERIFIER_VERSION, "seed": ev["seed"]}
            rec_path = os.path.join(out_dir, f"{tag}.{panel}.jsonl.gz")
            sum_path = os.path.join(out_dir, f"{tag}.{panel}.summary.json")
            if os.path.exists(sum_path) and os.path.exists(rec_path):
                old = json.load(open(sum_path))
                if old.get("adapter_sha256") == asha and old.get("protocol") == proto:
                    print(f"[xeval] {tag}/{panel}: reuse (same adapter hash and protocol)")
                    if res_dir:
                        _emit(res_dir, old, tag, panel)
                    continue
            seed = ev["seed"] * 1000003 + stable_int(f"{tag}:{panel}")
            torch.cuda.synchronize()
            t0 = time.time()
            recs, gen_s = run_attempts(model, tok, data[panel], fmt, ev["variant"], attempts, ev["cap"], seed,
                                       ev.get("batch_seqs", 512))
            wall = time.time() - t0
            for r in recs:
                r.update(tag=tag, panel=panel, adapter_sha256=asha, eval_seed=seed)
            write_jsonl_gz(rec_path, recs)
            s = summarize(recs)
            s.update(tag=tag, panel=panel, adapter=adapter, adapter_sha256=asha, protocol=proto, records=rec_path,
                     wall_s=wall, gen_s=gen_s, completions=len(recs),
                     generated_tokens=int(sum(r["n_tokens"] for r in recs)), gpu=torch.cuda.get_device_name(0),
                     counts=count_rows(recs, tag, panel, split))
            atomic_json(sum_path, s)
            ledger.flush(note=f"done {tag}/{panel}")
            print(f"[xeval] {tag}/{panel}: {len(recs)} completions wall={wall:.0f}s "
                  f"({len(recs) / wall:.1f}/s) tokens={s['generated_tokens']} gpu={s['gpu']}", flush=True)
            if res_dir:
                _emit(res_dir, s, tag, panel)
    ledger.flush(note="complete")
    if res_dir:
        atomic_json(os.path.join(res_dir, "summary.json"), {"run_id": cfg["run_id"], "gpu_s": ledger.total_gpu_s(),
                                                            "data": ev["data_name"]})
    return 0


def _emit(res_dir, s, tag, panel):
    """Compact distillate: throughput/summary JSON without per-problem rows, plus rows appended to xcounts.csv."""
    atomic_json(os.path.join(res_dir, f"xeval_{tag}.{panel}.json"),
                {k: v for k, v in s.items() if k not in ("records", "counts")})
    path = os.path.join(res_dir, "xcounts.csv")
    have = set()
    if os.path.exists(path):
        with open(path) as f:
            have = {(r["tag"], r["panel"]) for r in csv.DictReader(f)}
    if (tag, panel) in have:
        return
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=XCOLS)
        if new:
            w.writeheader()
        w.writerows(s["counts"])
