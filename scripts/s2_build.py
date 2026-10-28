#!/usr/bin/env python3
"""A6 program S, stage S2 (protocol Section 13.3): prescreen panels, then training pools and evaluation panels. CPU only.

Mode `prescreen`: from the st1 candidates (scripts/s2_data.py) write ext data `st1s` (xdata layout) with panels
  pre_pool  = the first --pool-per-class candidates of each class with role pool (first-draw order)
  pre_dev   = the first --panel-per-class candidates of each class with role dev_panel
  pre_conf  = the same for role conf_panel
  These are evaluated (xeval, independent attempts) under the light and extended seed-11 adapters. Prescreen outcomes
  only select and balance problems; they are never evaluation outcomes.
Mode `build`: from the candidates and the prescreen summaries write ext data `st1p` with panels
  trA, trB   training pools by coarsened exact matching (CEM) on: prescreen success bin under light and under
             extended, number-of-forms bin, target bin. Matched A/B pairs come from one cell; the pair order is
             shuffled (seeded) so every prefix of trA and trB stays matched cell by cell.
  sd3d, sd3f diagnostic panels (pre_dev / pre_conf): equal counts per class x mean-prescreen-rate bin.
  sr3d, sr3f representative panels: the first --rep candidates of role dev_panel / conf_panel in first-draw order that
             are not in a diagnostic panel (natural class mix; no prescreen selection).
  build_report.json: cell counts, unmatched cells, residual A - B differences of the matched pools, panel tables.
Usage: s2_build.py prescreen --config C --pool-per-class N --panel-per-class M
       s2_build.py build --config C --pre-tags pre-b16-s11,pre-b128-s11 --pool-max N --diag-per-cell K --rep R --out DIR
"""
import argparse
import gzip
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bc2rl import task  # noqa: E402

PROBLEM_KEYS = ("pid", "split", "n_inputs", "inputs", "target", "ref_trace", "structure", "ops_used")
SUCC_BINS = ((0, 0), (1, 7), (8, 15), (16, 16))            # prescreen successes of 16 attempts
FORM_BINS = ((1, 1), (2, 2), (3, 10**6))
TARGET_BINS = ((1, 60), (61, 200), (201, 1000))
RATE_BINS = ((0.0, 0.0), (1e-9, 0.25), (0.25 + 1e-9, 0.5), (0.5 + 1e-9, 0.75), (0.75 + 1e-9, 1 - 1e-9), (1.0, 1.0))


def binned(x, edges):
    for i, (lo, hi) in enumerate(edges):
        if lo <= x <= hi:
            return i
    raise ValueError(f"{x} outside {edges}")


def to_problem(c, split):
    d = {k: c[k] for k in PROBLEM_KEYS if k in c}
    d.update(split=split, n_inputs=3)
    return task.Problem(**d)


def write_ext(d, panels, extra):
    os.makedirs(d, exist_ok=True)
    man = {"panels": {}, **extra}
    for name, probs in panels.items():
        task.save_split(os.path.join(d, f"{name}.jsonl"), probs)
        man["panels"][name] = {"count": len(probs), "n_inputs": 3, "sha256": task.manifest_hash(probs),
                               "classes": dict(Counter(getattr(p, "_cls", None) or "?" for p in probs))}
    json.dump(man, open(os.path.join(d, "manifest.json.tmp"), "w"), indent=1)
    os.replace(os.path.join(d, "manifest.json.tmp"), os.path.join(d, "manifest.json"))
    return man


def select_prescreen(cands, pool_per_class, panel_per_class):
    out = {"pre_pool": [], "pre_dev": [], "pre_conf": []}
    want = {"pre_pool": ("pool", pool_per_class), "pre_dev": ("dev_panel", panel_per_class),
            "pre_conf": ("conf_panel", panel_per_class)}
    for name, (role, n) in want.items():
        for cl in ("A", "B"):
            out[name] += [c for c in cands if c["role"] == role and c["lab"]["cls"] == cl][:n]
        out[name].sort(key=lambda c: c["first_draw"])
    return out


def cem_pools(pre, succ, pool_max, seed):
    """pre: candidate dicts of pre_pool; succ: {pid: (c_light, c_extended)} of 16 attempts. Returns (trA, trB, report)."""
    cells = defaultdict(lambda: {"A": [], "B": []})
    for c in pre:
        if c["pid"] not in succ:
            continue
        cl, ce = succ[c["pid"]]
        key = (binned(cl, SUCC_BINS), binned(ce, SUCC_BINS), binned(c["lab"]["n_forms"], FORM_BINS),
               binned(c["target"], TARGET_BINS))
        cells[key][c["lab"]["cls"]].append(c)
    rng = random.Random(seed)
    pairs, table = [], []
    for key in sorted(cells):
        a, b = cells[key]["A"], cells[key]["B"]
        k = min(len(a), len(b))
        rng.shuffle(a)
        rng.shuffle(b)
        pairs += list(zip(a[:k], b[:k]))
        table.append({"cell": list(key), "A": len(a), "B": len(b), "matched": k})
    rng.shuffle(pairs)
    pairs = pairs[:pool_max]
    trA, trB = [p[0] for p in pairs], [p[1] for p in pairs]

    def props(cs):
        if not cs:
            return {}
        f = lambda g: sum(g(c) for c in cs) / len(cs)
        return {"n": len(cs), "succ_light": f(lambda c: succ[c["pid"]][0] / 16),
                "succ_extended": f(lambda c: succ[c["pid"]][1] / 16), "n_forms": f(lambda c: c["lab"]["n_forms"]),
                "n_traces": f(lambda c: c["lab"]["n_traces"]), "target": f(lambda c: c["target"]),
                "input_max": f(lambda c: c["lab"]["input_max"]), "input_sum": f(lambda c: c["lab"]["input_sum"]),
                "min_mult": f(lambda c: c["lab"]["min_mult"])}

    report = {"cells": table, "cells_without_support": [t["cell"] for t in table if t["matched"] == 0 and t["A"] + t["B"] > 0],
              "unmatched_A": sum(t["A"] - t["matched"] for t in table),
              "unmatched_B": sum(t["B"] - t["matched"] for t in table),
              "matched_pairs_total": sum(t["matched"] for t in table), "pool_size": len(trA),
              "pool_A": props(trA), "pool_B": props(trB),
              "pool_A_half": props(trA[:len(trA) // 2]), "pool_B_half": props(trB[:len(trB) // 2])}
    return trA, trB, report


def diag_panel(pre, succ, per_cell, seed):
    cells = defaultdict(list)
    for c in pre:
        if c["pid"] in succ:
            rate = (succ[c["pid"]][0] + succ[c["pid"]][1]) / 32
            cells[(c["lab"]["cls"], binned(rate, RATE_BINS))].append(c)
    rng = random.Random(seed)
    out, table = [], []
    for key in sorted(cells):
        cs = cells[key][:]
        rng.shuffle(cs)
        out += cs[:per_cell]
        table.append({"cls": key[0], "rate_bin": key[1], "available": len(cells[key]), "taken": min(per_cell, len(cs))})
    out.sort(key=lambda c: c["first_draw"])
    return out, table


def load_cands(cfg):
    from bc2rl.extension import ext_dir
    d = ext_dir(cfg, "st1")
    man = json.load(open(os.path.join(d, "manifest.json")))
    path = os.path.join(d, "candidates.jsonl.gz")
    if hashlib.sha256(open(path, "rb").read()).hexdigest() != man["candidates_sha256"]:
        raise SystemExit("st1 candidates hash mismatch")
    return [json.loads(x) for x in gzip.open(path, "rt")], man


def prescreen_succ(cfg, tags, panel):
    out = {}
    for t in tags:
        s = json.load(open(os.path.join(cfg["proj_data"], "evals", "ext", "st1s", f"{t}.{panel}.summary.json")))
        for r in s["counts"]:
            if int(r["n"]) != 16:
                raise SystemExit(f"{t}/{panel}/{r['pid']}: n = {r['n']} != 16")
            out.setdefault(r["pid"], []).append(int(r["c"]))
    return {k: tuple(v) for k, v in out.items() if len(v) == len(tags)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("prescreen", "build"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--pool-per-class", type=int)
    ap.add_argument("--panel-per-class", type=int)
    ap.add_argument("--pre-tags", default="pre-b16-s11,pre-b128-s11")
    ap.add_argument("--pool-max", type=int)
    ap.add_argument("--diag-per-cell", type=int)
    ap.add_argument("--rep", type=int)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--out")
    a = ap.parse_args()
    from bc2rl.extension import ext_dir
    from bc2rl.runtime import load_config
    cfg = load_config(a.config)
    cfg["proj_data"] = os.environ["PROJ_DATA"]
    cands, man = load_cands(cfg)
    if a.mode == "prescreen":
        d = ext_dir(cfg, "st1s")
        if os.path.exists(os.path.join(d, "manifest.json")):
            raise SystemExit(f"{d} exists; not overwritten")
        sel = select_prescreen(cands, a.pool_per_class, a.panel_per_class)
        panels = {k: [to_problem(c, k) for c in v] for k, v in sel.items()}
        m = write_ext(d, panels, {"source": "st1", "candidates_sha256": man["candidates_sha256"],
                                  "pool_per_class": a.pool_per_class, "panel_per_class": a.panel_per_class,
                                  "class_counts": {k: dict(Counter(c["lab"]["cls"] for c in v)) for k, v in sel.items()}})
        print(json.dumps({k: (v["count"], m["class_counts"][k]) for k, v in m["panels"].items()}))
        return
    tags = a.pre_tags.split(",")
    pre_man = json.load(open(os.path.join(ext_dir(cfg, "st1s"), "manifest.json")))
    by_pid = {c["pid"]: c for c in cands}
    pre = {p: [by_pid[x["pid"]] for x in map(json.loads, open(os.path.join(ext_dir(cfg, "st1s"), f"{p}.jsonl")))]
           for p in pre_man["panels"]}
    trA, trB, rep_pool = cem_pools(pre["pre_pool"], prescreen_succ(cfg, tags, "pre_pool"), a.pool_max, a.seed)
    sd3d, tab_d = diag_panel(pre["pre_dev"], prescreen_succ(cfg, tags, "pre_dev"), a.diag_per_cell, a.seed + 1)
    sd3f, tab_f = diag_panel(pre["pre_conf"], prescreen_succ(cfg, tags, "pre_conf"), a.diag_per_cell, a.seed + 2)
    used = {c["pid"] for c in sd3d + sd3f}
    sr3d = [c for c in cands if c["role"] == "dev_panel" and c["pid"] not in used][:a.rep]
    sr3f = [c for c in cands if c["role"] == "conf_panel" and c["pid"] not in used][:a.rep]
    sel = {"trA": trA, "trB": trB, "sd3d": sd3d, "sd3f": sd3f, "sr3d": sr3d, "sr3f": sr3f}
    panels = {}
    for k, v in sel.items():
        panels[k] = []
        for c in v:
            p = to_problem(c, k)
            p._cls = c["lab"]["cls"]
            panels[k].append(p)
    d = ext_dir(cfg, "st1p")
    if os.path.exists(os.path.join(d, "manifest.json")):
        raise SystemExit(f"{d} exists; not overwritten")
    report = {"pools": rep_pool, "diag_dev": tab_d, "diag_conf": tab_f, "pre_tags": tags, "seed": a.seed,
              "bins": {"succ": SUCC_BINS, "forms": FORM_BINS, "target": TARGET_BINS, "rate": RATE_BINS},
              "rep_class_shares": {k: dict(Counter(c["lab"]["cls"] for c in sel[k])) for k in ("sr3d", "sr3f")}}
    m = write_ext(d, panels, {"source": "st1", "candidates_sha256": man["candidates_sha256"], "report": report,
                              "class_of": {c["pid"]: c["lab"]["cls"] for v in sel.values() for c in v}})
    os.makedirs(a.out, exist_ok=True)
    json.dump({**report, "panels": {k: {"count": v["count"], "classes": v["classes"]} for k, v in m["panels"].items()}},
              open(os.path.join(a.out, "build_report.json"), "w"), indent=1)
    print(json.dumps({k: (v["count"], v["classes"]) for k, v in m["panels"].items()}))


if __name__ == "__main__":
    main()
