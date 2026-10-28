#!/usr/bin/env python3
"""A6 program S, stage S2 (protocol Section 13.3): 3-input candidates with exact structural labels. CPU only.

1. Universe: every (sorted input triple, target) that the frozen v1 generator can produce (generator rules: values in
   [min, max]; no zero subexpression; no * with an operand 1; |intermediate| <= bound; target in [1, max_target]).
   Class shares over the universe (uniform) are reported next to the generator-weighted shares.
2. Candidates: `draws` generator samples (3 inputs, seed 20261028). A key in any earlier split or panel is rejected
   and counted; a repeated st1 key is counted as a within-st1 duplicate. Each new key gets a role in first-draw order
   from a separate RNG: pool 1/2 (training pools, dev-visible), dev_panel 1/4, conf_panel 1/4 (fresh panels).
3. Labels from the complete solution set (`bc2rl.structure.labels`).
Writes $PROJ_DATA/data/v1/ext/st1/{candidates.jsonl.gz, manifest.json} once (an existing manifest is verified, never
overwritten) and s2_summary.json (small; distributions by role and class) to --out.
Usage: s2_data.py --config configs/data_v1.yaml --out DIR [--draws 400000] [--seed 20261028]
"""
import argparse
import gzip
import hashlib
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bc2rl import structure, task  # noqa: E402

ROLES = (("pool", 0.5), ("dev_panel", 0.25), ("conf_panel", 0.25))


def gen_config(cfg):
    g = cfg["data"]
    return task.GenConfig(min_value=g["min_value"], max_value=g["max_value"], max_target=g["max_target"],
                          max_abs_intermediate=g["max_abs_intermediate"], ops=tuple(g["ops"]))


def universe(gcfg):
    """Set of (sorted triple, target) that the generator can produce with 3 inputs."""
    out = set()
    vals = range(gcfg.min_value, gcfg.max_value + 1)

    def ok(op, a, b):
        if op == "*" and (a == 1 or b == 1):
            return None
        v = task.apply_op(op, a, b)
        return None if v == 0 or abs(v) > gcfg.max_abs_intermediate else v

    for a in vals:
        for b in vals:
            if b < a:
                continue
            for c in vals:
                if c < b:
                    continue
                tri = (a, b, c)
                for i, j, k in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
                    for x, y in ((tri[i], tri[j]), (tri[j], tri[i])):
                        for o1 in gcfg.ops:
                            r = ok(o1, x, y)
                            if r is None:
                                continue
                            for u, w in ((r, tri[k]), (tri[k], r)):
                                for o2 in gcfg.ops:
                                    t = ok(o2, u, w)
                                    if t is not None and 1 <= t <= gcfg.max_target:
                                        out.add((tri, t))
    return out


def draw_candidates(seed, draws, gcfg, seen, role_seed):
    """Returns (candidates in first-draw order, stats). `seen` (earlier keys) is not modified."""
    rng, rrng = random.Random(seed), random.Random(role_seed)
    st, gen_stats = Counter({"draws": 0, "excluded_existing": 0, "duplicate_within": 0}), Counter()
    by_key = {}
    cands = []
    for _ in range(draws):
        values, target, trace, struct, ops_used = task.sample_problem(rng, 3, gcfg, gen_stats)
        key = task.canonical_key(values, target)
        shown = values[:]
        rng.shuffle(shown)
        st["draws"] += 1
        if key in seen:
            st["excluded_existing"] += 1
            continue
        if key in by_key:
            st["duplicate_within"] += 1
            by_key[key]["draw_count"] += 1
            continue
        u, role = rrng.random(), None
        acc = 0.0
        for name, p in ROLES:
            acc += p
            if u < acc:
                role = name
                break
        c = {"pid": f"st1-3-{len(cands):06d}", "key": key, "inputs": shown, "target": target, "ref_trace": trace,
             "structure": struct, "ops_used": ops_used, "role": role or ROLES[-1][0], "draw_count": 1,
             "first_draw": st["draws"] - 1}
        by_key[key] = c
        cands.append(c)
    st["unique_new"] = len(cands)
    st.update({f"gen:{k}": v for k, v in gen_stats.items()})
    return cands, dict(st)


def q(xs, ps=(0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)):
    xs = sorted(xs)
    return [xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))] for p in ps] if xs else []


def binned(x, edges):
    for lo, hi in edges:
        if lo <= x <= hi:
            return f"{lo}-{hi}"
    return "other"


FORM_BINS = ((1, 1), (2, 2), (3, 4), (5, 10**6))
TARGET_BINS = ((1, 20), (21, 60), (61, 200), (201, 1000))


def summarize(cands):
    """Small distribution tables by role and class (labels must be present in each candidate as c['lab'])."""
    for c in cands:
        c.setdefault("lab", structure.labels(c["inputs"], c["target"]))
    groups = {"all": cands}
    for r, _ in ROLES:
        groups[r] = [c for c in cands if c["role"] == r]
    out = {"by_role_class": {}, "tables": {}}
    for g, cs in groups.items():
        out["by_role_class"][g] = dict(Counter(str(c["lab"]["cls"]) for c in cs))
        t = {}
        for cl in ("A", "B"):
            xs = [c for c in cs if c["lab"]["cls"] == cl]
            if not xs:
                continue
            t[cl] = {
                "n": len(xs),
                "draw_weighted_n": sum(c["draw_count"] for c in xs),
                "n_forms_q": q([c["lab"]["n_forms"] for c in xs]),
                "n_traces_q": q([c["lab"]["n_traces"] for c in xs]),
                "n_shapes_q": q([c["lab"]["n_shapes"] for c in xs]),
                "form_bins": dict(Counter(binned(c["lab"]["n_forms"], FORM_BINS) for c in xs)),
                "target_bins": dict(Counter(binned(c["target"], TARGET_BINS) for c in xs)),
                "target_q": q([c["target"] for c in xs]),
                "input_max_q": q([c["lab"]["input_max"] for c in xs]),
                "input_sum_q": q([c["lab"]["input_sum"] for c in xs]),
                "min_mult": dict(Counter(str(c["lab"]["min_mult"]) for c in xs)),
                "neg_needed": sum(c["lab"]["neg_needed"] for c in xs),
                "zero_needed": sum(c["lab"]["zero_needed"] for c in xs),
                "gen_structure": dict(Counter(c["structure"] for c in xs)),
                "top_shapes": Counter(s for c in xs for s in c["lab"]["shapes"]).most_common(15),
                "cross_form_target": dict(Counter(f"{binned(c['lab']['n_forms'], FORM_BINS)}|{binned(c['target'], TARGET_BINS)}"
                                                  for c in xs)),
            }
        out["tables"][g] = t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=20261028)
    a = ap.parse_args()
    from bc2rl.extension import existing_keys, ext_dir
    from bc2rl.runtime import load_config
    cfg = load_config(a.config)
    cfg["proj_data"] = os.environ["PROJ_DATA"]
    gcfg = gen_config(cfg)
    d = ext_dir(cfg, "st1")
    root = os.path.dirname(d)
    ext_names = sorted(n for n in os.listdir(root) if n != "st1" and os.path.exists(os.path.join(root, n, "manifest.json")))
    bad = [n for n in ext_names if "panels" not in json.load(open(os.path.join(root, n, "manifest.json")))]
    if bad:
        raise SystemExit(f"ext data without a panels manifest (cannot exclude its keys): {bad}")
    seen = existing_keys(cfg, ext_names)
    seen3 = sum(len(k[0]) == 3 for k in seen)
    cpath, mpath = os.path.join(d, "candidates.jsonl.gz"), os.path.join(d, "manifest.json")
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(mpath):
        man = json.load(open(mpath))
        if hashlib.sha256(open(cpath, "rb").read()).hexdigest() != man["candidates_sha256"]:
            raise SystemExit("st1 candidates hash mismatch")
        cands = [json.loads(x) for x in gzip.open(cpath, "rt")]
        print(f"[s2] st1 exists; hash OK; {len(cands)} candidates")
    else:
        cands, stats = draw_candidates(a.seed, a.draws, gcfg, seen, role_seed=a.seed + 1)
        for c in cands:
            c["lab"] = structure.labels(c["inputs"], c["target"])
            c["key"] = [list(c["key"][0]), c["key"][1]]
        uni = universe(gcfg)
        uni_cls = Counter(str(structure.labels(list(t), y)["cls"]) for t, y in uni)
        os.makedirs(d, exist_ok=True)
        tmp = cpath + ".tmp"
        with gzip.open(tmp, "wt") as f:
            for c in cands:
                f.write(json.dumps(c) + "\n")
        os.replace(tmp, cpath)
        man = {"name": "st1", "seed": a.seed, "role_seed": a.seed + 1, "roles": dict(ROLES), "draws": a.draws,
               "generator": task.GENERATOR_VERSION, "gen_config": cfg["data"], "verifier": task.VERIFIER_VERSION,
               "exclude_ext": ext_names, "existing_keys": len(seen), "existing_keys_3input": seen3, "stats": stats,
               "universe": {"n": len(uni), "cls": dict(uni_cls)},
               "candidates_sha256": hashlib.sha256(open(cpath, "rb").read()).hexdigest()}
        json.dump(man, open(mpath + ".tmp", "w"), indent=1)
        os.replace(mpath + ".tmp", mpath)
    summ = summarize(cands)
    summ["manifest"] = man
    json.dump(summ, open(os.path.join(a.out, "s2_summary.json"), "w"), indent=1)
    print(f"[s2] candidates={len(cands)} classes={summ['by_role_class']['all']} universe={man['universe']}")
    print(f"[s2] stats={man['stats']}")


if __name__ == "__main__":
    main()
