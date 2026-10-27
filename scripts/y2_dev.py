"""A6 development analysis (protocol/AMENDMENT-A6-2026-10-26.md Sections 5.1 and 8.2). Exploratory: yd panels only.

Usage: python scripts/y2_dev.py --evals $PROJ_DATA/evals/ext/a6 --config configs/a6/ydata.json --out DIR
Writes y2_dev.json (Delta at caps 256 / 1024, K = 1 code decomposition, truncation rescue, lenient parse,
structure), y2_adapters.csv (per-adapter rates) and y4_matching.json (Section 8.2 profiles and matched pairs).
Refuses to run unless every required yd adapter is present.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations

sys.path.insert(0, os.getcwd())
from bc2rl import yanalysis as Y  # noqa: E402
from bc2rl.extension import load_ext  # noqa: E402
from bc2rl.runtime import load_config, read_jsonl_gz  # noqa: E402
from bc2rl.xanalysis import boot_idx  # noqa: E402

SEEDS = (11, 22, 33, 44, 55, 66)
METHODS = ("grpo", "filtered")
STEPS = (16, 32, 64, 128)


def it(n, s):
    return f"b{n}-s{s}"


def ft(n, m, s, arm="M"):
    return f"b{n}-{m}-{arm}-s{s}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = load_config(a.config)
    probs, _ = load_ext(cfg, "a6")
    c3, c5 = Y.load_summaries(a.evals, "yd3"), Y.load_summaries(a.evals, "yd5")
    inits = [it(n, s) for n in STEPS for s in SEEDS]
    finals = [ft(n, m, s) for n in (16, 128) for m in METHODS for s in SEEDS]
    missing = [t for t in inits + finals if t not in c3] + [t for t in inits if t not in c5]
    if missing:
        raise SystemExit(f"yd incomplete, missing {missing}; analysis refused")
    ev = Y.Evals({t: c3[t] for t in inits + finals})
    ev5 = Y.Evals({t: c5[t] for t in inits})
    idx = boot_idx(len(ev.pids), a.n_boot, 6)
    w = Y.pooled_weights(ev, inits)
    pairs = {(m, s): (it(16, s), ft(16, m, s), it(128, s), ft(128, m, s)) for m in METHODS for s in SEEDS}
    unit = lambda k: k[1]
    res = {"weights": w.tolist(), "n_problems": {"yd3": len(ev.pids), "yd5": len(ev5.pids)}}
    # Delta at both caps (subgroup fixed by cap-256 selection), and their paired difference
    D = {}
    for cap in (256, 1024):
        summ, P, R = Y.init_contrast(ev, pairs, w, idx, unit, cap=cap)
        res[f"delta_cap{cap}"] = summ
        D[cap] = (P, R)
    res["delta_256_minus_1024"] = Y.diff_of(*D[256], *D[1024], unit, idx)
    # K = 1 code decomposition at both caps (sum over codes = Delta at K = 1)
    for cap in (256, 1024):
        res[f"delta_K1_cap{cap}"] = Y.init_contrast(ev, pairs, w, idx, unit, K=1, cap=cap)[0]
        res[f"decomposition_K1_cap{cap}"] = {
            code: Y.init_contrast(ev, pairs, w, idx, unit, K=1, cap=cap, code=code)[0] for code in Y.FAIL_CODES}
    # per-adapter rates from the summaries; rescue of truncated attempts from the records
    rows = []
    rescue = {}
    for t in inits + finals:
        for pname, src in (("yd3", c3), ("yd5", c5)):
            if t not in src:
                continue
            s = json.load(open(os.path.join(a.evals, f"{t}.{pname}.summary.json")))
            for h, e in s["ext"].items():
                rows.append({"tag": t, "panel": pname, "n_inputs": h, "pass1_256": s["by_inputs"][h]["pass@1"]
                             if "pass@1" in s["by_inputs"][h] else None, **{k: v for k, v in e.items()
                                                                            if not isinstance(v, dict)},
                             **{f"c256_{k}": v for k, v in e["codes_256"].items()},
                             **{f"cx_{k}": v for k, v in e["codes_x"].items()}})
        recs = read_jsonl_gz(os.path.join(a.evals, f"{t}.yd3.jsonl.gz"))
        tr = [r for r in recs if r["finish"] == "length"]
        rescue[t] = {"truncated_256": len(tr), "attempts": len(recs),
                     "outcome_at_1024": dict(Counter(r["x_code"] for r in tr)),
                     "strict_fail_lenient_success_x": sum(int(r.get("x_lenient", 0)) for r in recs)}
    res["rescue_yd3"] = rescue
    # structure and strategy on yd3 (exploratory)
    forms = {}
    for t in inits + finals:
        fm = defaultdict(Counter)
        for r in read_jsonl_gz(os.path.join(a.evals, f"{t}.yd3.jsonl.gz")):
            if r["reward"]:
                fm[r["pid"]][Y.form_of_text(r["text"])] += 1
        forms[t] = fm
    res["structure"] = Y.structure_analysis(ev, probs["yd3"], forms, [
        {"init": f"b{n}", "method": m, "seed": s, "ti": it(n, s), "tf": ft(n, m, s)}
        for n in (16, 128) for m in METHODS for s in SEEDS])
    json.dump(res, open(os.path.join(a.out, "y2_dev.json"), "w"), indent=1, default=float)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(a.out, "y2_adapters.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    # Y4 matching (Section 8.2): profiles of the 24 initial adapters; all pairs with different BC steps
    prof = {t: Y.competence_profile(ev, ev5, t) for t in inits}
    matched, cand = [], []
    for x, y in combinations(inits, 2):
        nx, ny = int(x.split("-")[0][1:]), int(y.split("-")[0][1:])
        if nx == ny:
            continue
        lo, hi = (x, y) if nx < ny else (y, x)
        ok, r = Y.match(prof[lo], prof[hi])
        cand.append({"low": lo, "high": hi, "matched": ok, **r})
        if ok:
            matched.append({"low": lo, "high": hi, **r})
    json.dump({"rule": "AMENDMENT-A6 Section 8.2", "profiles": prof, "matched_pairs": matched, "all_pairs": cand},
              open(os.path.join(a.out, "y4_matching.json"), "w"), indent=1, default=float)
    print(f"[y2-dev] Delta cap256 {res['delta_cap256']['point']:+.4f} cap1024 {res['delta_cap1024']['point']:+.4f}; "
          f"matched pairs {len(matched)}", flush=True)


if __name__ == "__main__":
    main()
