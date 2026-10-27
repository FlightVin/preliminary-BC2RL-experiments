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
import re
import time
from collections import Counter, defaultdict

import numpy as np
import torch

from . import task
from .data import data_dir
from .evaluate import run_attempts, stable_int, summarize
from .mech import MECH_SPLITS, mech_dir
from .modeling import DECODING_RECORD, continue_sampling, load_adapter, load_policy, load_tokenizer, model_info
from .runtime import CostLedger, atomic_json, write_jsonl_gz


def ext_dir(cfg, name):
    return os.path.join(data_dir(cfg), "ext", name)


def existing_keys(cfg, exclude_ext=()):
    """Canonical keys of every v1 split, the format example, the mechanism splits and the named ext panels."""
    d = data_dir(cfg)
    man = json.load(open(os.path.join(d, "manifest.json")))
    seen = set()
    for s in list(man["splits"]) + ["format"]:
        seen.update(p.key for p in task.load_split(os.path.join(d, f"{s}.jsonl")))
    for s in MECH_SPLITS:
        seen.update(p.key for p in task.load_split(os.path.join(mech_dir(cfg), f"{s}.jsonl")))
    for name in exclude_ext:
        probs, _ = load_ext(cfg, name)
        for ps in probs.values():
            seen.update(p.key for p in ps)
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
    seen = existing_keys(cfg, xd.get("exclude_ext", ()))
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
           "existing_keys_excluded": n_existing, "exclude_ext": list(xd.get("exclude_ext", ())), "rejection_stats": {k: dict(v) for k, v in stats.items()},
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


# A6 failure codes (protocol/AMENDMENT-A6 Section 3.3): one letter per attempt
CODE = {"success": "S", "truncated": "T", "malformed": "F", "no_ops": "F", "illegal_operand": "I",
        "arith_error": "A", "too_few_ops": "P", "wrong_final": "W"}
_LENIENT = re.compile(r"(-?\d+)\s*([+\-*×xX])\s*(-?\d+)\s*=\s*(-?\d+)")


def lenient_success(inputs, target, text):
    """Diagnostic only (never a reward): every `a op b = c` substring, in order, checked as a chain; also the last
    n - 1 such substrings. 1 if either chain is valid and reaches the target."""
    eqs = _LENIENT.findall(text.replace("\u2212", "-"))
    lines = [f"{a} {'*' if o in 'xX×' else o} {b} = {c}" for a, o, b, c in eqs]
    k = len(inputs) - 1
    for cand in (lines, lines[-k:] if len(lines) > k else None):
        if cand and task.verify(inputs, target, "\n".join(cand))["reward"]:
            return 1
    return 0


def extend_records(model, tok, recs, probs_by_pid, cap, cap_ext, seed, batch_seqs=256):
    """Cap extension: continue every attempt truncated at `cap` from its own prefix up to `cap_ext` tokens and add
    the cap-`cap_ext` outcome (x_* fields). Attempts that stopped before `cap` keep their outcome. Needs the
    `_prompt_ids` / `_completion_ids` fields from `run_attempts(..., keep_ids=True)`."""
    from .modeling import eos_ids
    stops = set(eos_ids(tok))
    todo = [k for k, r in enumerate(recs) if r["finish"] == "length"]
    cont = continue_sampling(model, tok, [(recs[k]["_prompt_ids"], recs[k]["_completion_ids"]) for k in todo],
                             cap_ext - cap, seed + 7919, batch_seqs) if todo else []
    for r in recs:
        r["code"] = CODE[r["error"]]
        if r["finish"] != "length":
            r.update(x_reward=r["reward"], x_error=r["error"], x_code=r["code"], x_finish=r["finish"],
                     x_n_tokens=r["n_tokens"])
    for k, (new, fin) in zip(todo, cont):
        r = recs[k]
        p = probs_by_pid[r["pid"]]
        ids = list(r["_completion_ids"]) + list(new)
        text = tok.decode([t for t in ids if t not in stops], skip_special_tokens=True)
        v = task.verify(p.inputs, p.target, text, truncated=(fin == "length"))
        r.update(x_text=text, x_reward=v["reward"], x_error=v["error"], x_code=CODE[v["error"]], x_finish=fin,
                 x_n_tokens=len(ids))
    for r in recs:
        if not r["x_reward"]:
            p = probs_by_pid[r["pid"]]
            r["x_lenient"] = lenient_success(p.inputs, p.target, r.get("x_text", r["text"]))
    return len(todo)


def count_rows(recs, tag, panel, split):
    by = defaultdict(dict)
    meta = {}
    for r in recs:
        by[r["pid"]][int(r["attempt"])] = r
        meta[r["pid"]] = int(r["n_inputs"])
    rows = []
    for pid in sorted(by):
        a = by[pid]
        n = len(a)
        if sorted(a) != list(range(n)):
            raise SystemExit(f"{tag}/{pid}: attempts not contiguous")
        rew = [int(a[i]["reward"]) for i in range(n)]
        row = {"tag": tag, "panel": panel, "pid": pid, "n_inputs": meta[pid], "n": n, "c": sum(rew),
               "c_a": sum(rew[:split]) if split else "", "c_b": sum(rew[split:]) if split else "",
               "mean_tokens": round(float(np.mean([a[i]["n_tokens"] for i in range(n)])), 1),
               "n_trunc": sum(a[i]["finish"] == "length" for i in range(n))}
        if "x_reward" in a[0]:
            xr = [int(a[i]["x_reward"]) for i in range(n)]
            row.update({"c_x": sum(xr), "c_a_x": sum(xr[:split]) if split else "",
                        "c_b_x": sum(xr[split:]) if split else "",
                        "n_trunc_x": sum(a[i]["x_finish"] == "length" for i in range(n)),
                        "mean_tokens_x": round(float(np.mean([a[i]["x_n_tokens"] for i in range(n)])), 1),
                        "codes": "".join(a[i]["code"] for i in range(n)),
                        "x_codes": "".join(a[i]["x_code"] for i in range(n)),
                        "n_lenient_x": sum(int(a[i].get("x_lenient", 0)) for i in range(n))})
        rows.append(row)
    return rows


def ext_summary(recs):
    """Cap-extension rates per input count: success, truncation, code shares and lenient successes at cap_ext."""
    out = {}
    for h in sorted({r["n_inputs"] for r in recs}):
        rs = [r for r in recs if r["n_inputs"] == h]
        n = len(rs)
        out[str(h)] = {"pass1_x": sum(r["x_reward"] for r in rs) / n,
                       "trunc_256": sum(r["finish"] == "length" for r in rs) / n,
                       "trunc_x": sum(r["x_finish"] == "length" for r in rs) / n,
                       "codes_256": dict(Counter(r["code"] for r in rs)),
                       "codes_x": dict(Counter(r["x_code"] for r in rs)),
                       "rescued": sum(r["finish"] == "length" and r["x_reward"] == 1 for r in rs),
                       "lenient_x": sum(int(r.get("x_lenient", 0)) for r in rs),
                       "mean_tokens_x": float(np.mean([r["x_n_tokens"] for r in rs]))}
    return out


XCOLS = ["tag", "panel", "pid", "n_inputs", "n", "c", "c_a", "c_b", "mean_tokens", "n_trunc"]


def cmd_xeval(cfg):
    """Evaluate each checkpoint on each panel; skip (tag, panel) pairs whose summary matches adapter hash + protocol."""
    ev = cfg["xeval"]
    pd = cfg["proj_data"]
    data, man = load_ext(cfg, ev["data_name"])
    fmt = task.load_split(os.path.join(data_dir(cfg), "format.jsonl"))[0]
    info = model_info(pd)
    tok = load_tokenizer(info["path"])
    model = load_policy(cfg.get("model_path_override") or info["path"], cfg["model"]["lora"], seed=0,
                        device=cfg.get("device", "cuda"))
    out_dir = os.path.join(pd, "evals", "ext", ev["data_name"])
    os.makedirs(out_dir, exist_ok=True)
    ledger = CostLedger(os.path.join(out_dir, f"cost-{cfg['run_id']}.json"), "xeval")
    res_dir = cfg.get("_results_dir")
    cap_ext = ev.get("cap_ext")
    emit_counts = ev.get("emit_counts", True)
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
            if cap_ext:
                proto["cap_ext"] = cap_ext
            rec_path = os.path.join(out_dir, f"{tag}.{panel}.jsonl.gz")
            sum_path = os.path.join(out_dir, f"{tag}.{panel}.summary.json")
            if os.path.exists(sum_path) and os.path.exists(rec_path):
                old = json.load(open(sum_path))
                if old.get("adapter_sha256") == asha and old.get("protocol") == proto:
                    print(f"[xeval] {tag}/{panel}: reuse (same adapter hash and protocol)")
                    if res_dir:
                        _emit(res_dir, old, tag, panel, emit_counts)
                    continue
            seed = ev["seed"] * 1000003 + stable_int(f"{tag}:{panel}")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            recs, gen_s = run_attempts(model, tok, data[panel], fmt, ev["variant"], attempts, ev["cap"], seed,
                                       ev.get("batch_seqs", 512), keep_ids=bool(cap_ext))
            n_cont, ext_s = 0, 0.0
            if cap_ext:
                t1 = time.time()
                n_cont = extend_records(model, tok, recs, {p.pid: p for p in data[panel]}, ev["cap"], cap_ext, seed,
                                        ev.get("ext_batch_seqs", 256))
                ext_s = time.time() - t1
                for r in recs:
                    r.pop("_prompt_ids", None)
                    r.pop("_completion_ids", None)
            wall = time.time() - t0
            for r in recs:
                r.update(tag=tag, panel=panel, adapter_sha256=asha, eval_seed=seed)
            write_jsonl_gz(rec_path, recs)
            s = summarize(recs)
            s.update(tag=tag, panel=panel, adapter=adapter, adapter_sha256=asha, protocol=proto, records=rec_path,
                     wall_s=wall, gen_s=gen_s, completions=len(recs),
                     generated_tokens=int(sum(r["n_tokens"] for r in recs)),
                     gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                     counts=count_rows(recs, tag, panel, split))
            if cap_ext:
                s.update(ext=ext_summary(recs), n_continued=n_cont, ext_s=ext_s)
            atomic_json(sum_path, s)
            ledger.flush(note=f"done {tag}/{panel}")
            print(f"[xeval] {tag}/{panel}: {len(recs)} completions wall={wall:.0f}s "
                  f"({len(recs) / wall:.1f}/s) tokens={s['generated_tokens']} gpu={s['gpu']}", flush=True)
            if res_dir:
                _emit(res_dir, s, tag, panel, emit_counts)
    ledger.flush(note="complete")
    if res_dir:
        atomic_json(os.path.join(res_dir, "summary.json"), {"run_id": cfg["run_id"], "gpu_s": ledger.total_gpu_s(),
                                                            "data": ev["data_name"]})
    return 0


def _emit(res_dir, s, tag, panel, emit_counts=True):
    """Compact distillate: throughput/summary JSON without per-problem rows, plus rows appended to xcounts.csv
    (A6 sets emit_counts false: its per-problem rows stay on PROJ_DATA and the analysis jobs read them there)."""
    atomic_json(os.path.join(res_dir, f"xeval_{tag}.{panel}.json"),
                {k: v for k, v in s.items() if k not in ("records", "counts")})
    if not emit_counts:
        return
    path = os.path.join(res_dir, "xcounts.csv")
    have = set()
    if os.path.exists(path):
        with open(path) as f:
            have = {(r["tag"], r["panel"]) for r in csv.DictReader(f)}
    if (tag, panel) in have:
        return
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=XCOLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(s["counts"])
