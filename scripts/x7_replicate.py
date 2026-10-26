"""Pre-registered training-replication analysis X7 (protocol/AMENDMENT-A5-2026-10-26.md Section 9.3); runs once on
the complete set of new-seed adapters. Uses the X5 estimator unchanged, with the seed set replaced by the new seeds.

Writes to --out: x7_primary.json (primary + decision + sensitivities + seed-level and 6-seed descriptive results),
x7_bins.csv, x7_contrasts.csv (whole-panel final - initial, both panels).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from bc2rl import xanalysis as X  # noqa: E402

NEW_SEEDS = (44, 55, 66)
OLD_SEEDS = (11, 22, 33)
T975_DF5 = 2.5706                     # Student t quantile, 0.975, 5 degrees of freedom


def seed_deltas(D):
    """Delta_s = mean over methods of the usable Delta(m, s); a seed without a usable pair is NaN (counts as not > 0)."""
    out = {}
    for s in X.SEEDS:
        v = [D[(m, s)] for m in X.METHODS if (m, s) in D and np.isfinite(D[(m, s)])]
        out[s] = float(np.mean(v)) if v else float("nan")
    return out


def decide_x7(lo, hi, ds):
    if lo > 0 and all(np.isfinite(v) and v > 0 for v in ds.values()):
        return "REPLICATED"
    if hi < 0:
        return "CONTRADICTED"
    return "NOT REPLICATED"


def analyse(panel, n_boot, seeds):
    X.SEEDS = tuple(seeds)
    prim = X.primary(panel, n_boot=n_boot)
    pt = X.comparison(panel)
    ds = seed_deltas(pt["D"])
    return prim, ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    X.SEEDS = NEW_SEEDS
    panels = {}
    for name in ("xc3", "xc5"):
        counts = X.load_counts(a.evals, name)
        missing = [t for t in X.required_tags() if t not in counts]
        if missing:
            raise SystemExit(f"{name}: missing adapters {missing}; analysis refused")
        panels[name] = X.Panel({t: counts[t] for t in X.required_tags()})
    p3 = panels["xc3"]
    prim, ds = analyse(p3, a.n_boot, NEW_SEEDS)
    res = {"primary": prim, "seed_deltas": {str(k): v for k, v in ds.items()},
           "decision": decide_x7(prim["ci95"][0], prim["ci95"][1], ds)}
    print(f"[x7] PRIMARY mean {prim['mean_delta']:+.4f} {prim['ci95']} seeds {ds} -> {res['decision']}", flush=True)
    sens = {"swap_roles": dict(swap=True), "K4": dict(K=4), "K8": dict(K=8), "K24": dict(K=24),
            "consistency_rel16": dict(measure="rel")}
    res["sensitivity"] = {k: X.primary(p3, n_boot=a.n_boot, **v) for k, v in sens.items()}
    for k, v in res["sensitivity"].items():
        print(f"[x7] {k}: {v['mean_delta']:+.4f} {v['ci95']}", flush=True)
    rows = X.per_bin_table(p3)
    contr = []
    for name, p in panels.items():
        f, c = X.full_panel(p, idx=X.boot_idx(len(p.pids), a.n_boot, 2))
        contr += [dict(panel=name, **r) for r in c]
    # 6-seed descriptive (secondary): seed-level mean and t interval over 11, 22, 33 (X5 adapters) and the new seeds
    counts = X.load_counts(a.evals, "xc3")
    X.SEEDS = OLD_SEEDS
    old_tags = X.required_tags()
    if all(t in counts for t in old_tags):
        pold = X.Panel({t: counts[t] for t in old_tags})
        ds_old = seed_deltas(X.comparison(pold)["D"])
        v = np.array(list(ds_old.values()) + list(ds.values()))
        m, se = float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))
        res["six_seed"] = {"seed_deltas": {str(k): x for k, x in {**ds_old, **ds}.items()}, "mean": m,
                           "t95": [m - T975_DF5 * se, m + T975_DF5 * se], "n_positive": int((v > 0).sum())}
        print(f"[x7] six-seed mean {m:+.4f} t95 {res['six_seed']['t95']}", flush=True)
    X.SEEDS = NEW_SEEDS
    res["n_problems"] = {k: len(v.pids) for k, v in panels.items()}
    res["n_boot"] = a.n_boot
    json.dump(res, open(os.path.join(a.out, "x7_primary.json"), "w"), indent=1, default=float)
    import csv
    for fn, rr in (("x7_bins.csv", rows), ("x7_contrasts.csv", contr)):
        if rr:
            keys = list(dict.fromkeys(k for r in rr for k in r))
            with open(os.path.join(a.out, fn), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(rr)
    print("[x7] done", flush=True)


if __name__ == "__main__":
    main()
