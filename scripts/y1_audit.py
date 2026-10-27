"""A6 stage Y1 (protocol/AMENDMENT-A6-2026-10-26.md Section 4): interpretation audit of the A5 result on the existing
X5/X7 evaluations (xc3, xc5; seeds 11-66). Exploratory / audit only: these panels served the A5 confirmations.

Usage: python scripts/y1_audit.py --evals $PROJ_DATA/evals/ext/x5 --config configs/ext/xdata.json --out DIR
Writes y1_audit.json (items 1-5, 6 summary, lenient parse), y1_pairs.csv, y1_decomp.csv, y1_npmle.csv,
y1_structure.json.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.getcwd())
from bc2rl import yanalysis as Y  # noqa: E402
from bc2rl.extension import lenient_success, load_ext  # noqa: E402
from bc2rl.runtime import load_config, read_jsonl_gz  # noqa: E402
from bc2rl.xanalysis import PRIMARY_BINS, boot_idx, ci, loss_k  # noqa: E402

SEEDS = (11, 22, 33, 44, 55, 66)
METHODS = ("grpo", "filtered")
INITS = ("light", "extended")


def it(i, s):
    return f"{i}-s{s}"


def ft(i, m, s):
    return f"{i}-{m}-s{s}-r256"


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def pairs_for(seeds):
    return {(m, s): (it("light", s), ft("light", m, s), it("extended", s), ft("extended", m, s))
            for m in METHODS for s in seeds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = load_config(a.config)
    probs, _ = load_ext(cfg, "x5")
    tags = [it(i, s) for i in INITS for s in SEEDS] + [ft(i, m, s) for i in INITS for m in METHODS for s in SEEDS]
    # rows with cap-256 codes from the full records; texts kept only for successes and strict failures (lenient)
    counts, forms, lenient = {}, {}, {}
    for t in tags:
        split = 24 if t.count("-") == 1 else 0
        recs = read_jsonl_gz(os.path.join(a.evals, f"{t}.xc3.jsonl.gz"))
        counts[t] = Y.counts_from_records(recs, split)
        pmap = {p.pid: p for p in probs["xc3"]}
        fm = defaultdict(Counter)
        nlen = 0
        for r in recs:
            if r["reward"]:
                fm[r["pid"]][Y.form_of_text(r["text"])] += 1
            elif lenient_success(pmap[r["pid"]].inputs, pmap[r["pid"]].target, r["text"]):
                nlen += 1
        forms[t] = fm
        lenient[t] = {"strict_failures": sum(1 for r in recs if not r["reward"]), "lenient_successes": nlen,
                      "attempts": len(recs)}
    ev = Y.Evals(counts, tags)
    c5 = Y.load_summaries(a.evals, "xc5")
    ev5 = Y.Evals({t: c5[t] for t in tags}, tags)
    idx = boot_idx(len(ev.pids), a.n_boot, 6)
    res = {"n_problems": len(ev.pids), "n_boot": a.n_boot, "lenient": lenient}
    unit = lambda k: k[1]                                   # seed
    # 0. reproduce the A5 primaries (weights pooled within each seed set)
    res["a5_reproduction"] = {}
    for name, seeds in (("x5", SEEDS[:3]), ("x7", SEEDS[3:])):
        w = Y.pooled_weights(ev, [it(i, s) for i in INITS for s in seeds])
        summ, P, _ = Y.init_contrast(ev, pairs_for(seeds), w, None, lambda k: k)
        res["a5_reproduction"][name] = {"mean": summ["point"], "pairs": {f"{k[0]}-s{k[1]}": v for k, v in P.items()}}
    # common weights for the audit: pooled over all 12 initial adapters
    w = Y.pooled_weights(ev, [it(i, s) for i in INITS for s in SEEDS])
    res["weights_pooled12"] = w.tolist()
    pairs = pairs_for(SEEDS)
    # 1. change versus final state, K = 16 and K = 1
    res["change_vs_final"] = {}
    pair_rows = []
    for K in (16, 1):
        out = {}
        for meas in ("excess", "final", "base"):
            summ, P, R = Y.init_contrast(ev, pairs, w, idx, unit, K=K, measure=meas)
            out[meas] = summ
            for k, v in P.items():
                pair_rows.append({"K": K, "measure": meas, "method": k[0], "seed": k[1], "delta": v,
                                  "boot_lo": ci(R[k])[0], "boot_hi": ci(R[k])[1]})
        res["change_vs_final"][f"K{K}"] = out
    # whole-panel final and initial levels: extended - light, pass@1 and pass@16
    wp = []
    for pname, e in (("xc3", ev), ("xc5", ev5)):
        for K in (1, 16):
            for m in METHODS:
                for s in SEEDS:
                    fl = 1 - loss_k(32, e.arr(ft("light", m, s), "c"), K)
                    fe = 1 - loss_k(32, e.arr(ft("extended", m, s), "c"), K)
                    d = fe - fl
                    wp.append({"panel": pname, "k": K, "method": m, "seed": s, "final_light": float(fl.mean()),
                               "final_ext": float(fe.mean()), "final_ext_minus_light": float(d.mean()),
                               "boot_lo": ci(d[boot_idx(len(d), 2000, 7)].mean(1))[0],
                               "boot_hi": ci(d[boot_idx(len(d), 2000, 7)].mean(1))[1]})
            for s in SEEDS:
                il = 1 - loss_k(48, e.arr(it("light", s), "c"), K)
                ie = 1 - loss_k(48, e.arr(it("extended", s), "c"), K)
                wp.append({"panel": pname, "k": K, "method": "initial", "seed": s, "final_light": float(il.mean()),
                           "final_ext": float(ie.mean()), "final_ext_minus_light": float((ie - il).mean())})
    res["whole_panel_rows"] = len(wp)
    # 2. selection and weighting
    wsets = {"pooled": w, "light_only": Y.pooled_weights(ev, [it("light", s) for s in SEEDS]),
             "extended_only": Y.pooled_weights(ev, [it("extended", s) for s in SEEDS]),
             "equal": np.ones(3) / 3}
    res["weighting"] = {k: {"w": v.tolist(), **Y.init_contrast(ev, pairs, v, idx, unit)[0]} for k, v in wsets.items()}
    res["per_bin"] = {}
    for j, (lo, hi) in enumerate(PRIMARY_BINS):
        P, R = {}, {}
        for k, (li, lf, hi_, hf) in pairs.items():
            if j not in Y.usable(ev, [li, hi_]):
                continue
            a_, ar = Y.branch(ev, li, lf, w, [j], idx)
            b_, br = Y.branch(ev, hi_, hf, w, [j], idx)
            P[k], R[k] = b_ - a_, br - ar
        res["per_bin"][f"{lo}-{hi}"] = Y.group_units(P, R, unit, idx)
        P2, R2 = {}, {}
        for k, (li, lf, hi_, hf) in pairs.items():
            use = [u for u in Y.usable(ev, [li, hi_]) if u != j]
            a_, ar = Y.branch(ev, li, lf, w, use, idx)
            b_, br = Y.branch(ev, hi_, hf, w, use, idx)
            P2[k], R2[k] = b_ - a_, br - ar
        res["per_bin"][f"without {lo}-{hi}"] = Y.group_units(P2, R2, unit, idx)
    # 3. seed dependence: two-level bootstrap (seeds, then problems)
    _, P, R = Y.init_contrast(ev, pairs, w, idx, unit)
    seed_pts = {s: np.mean([P[(m, s)] for m in METHODS]) for s in SEEDS}
    seed_reps = {s: np.mean([R[(m, s)] for m in METHODS], axis=0) for s in SEEDS}
    rng = np.random.default_rng(8)
    two = []
    for b in range(a.n_boot):
        ss = rng.choice(SEEDS, size=len(SEEDS))
        two.append(np.mean([seed_reps[s][b] for s in ss]))
    res["seed_dependence"] = {"pairs": {f"{k[0]}-s{k[1]}": v for k, v in P.items()},
                              "seed_values": {str(s): float(v) for s, v in seed_pts.items()},
                              "two_level_boot95": list(ci(np.array(two))),
                              "summary": Y.summarize_contrast(seed_pts, seed_reps)}
    # 4. observed counts are not true probabilities: NPMLE per initial adapter
    npm = []
    for i in INITS:
        for s in SEEDS:
            t = it(i, s)
            p, pi = Y.npmle(ev.arr(t, "c"), 48)
            post = Y.posterior_mean_by_bin(p, pi, 24)
            b = Y.bins_for(ev, t)
            cb = ev.arr(t, "c_b")
            for j, (lo, hi) in enumerate(PRIMARY_BINS):
                npm.append({"init": i, "seed": s, "bin": f"{lo}-{hi}", "n": int((b == j).sum()),
                            "model_E_p": post[j], "observed_base_rate": float(cb[b == j].mean() / 24),
                            "sel_rate": float(ev.arr(t, "c_a")[b == j].mean() / 24),
                            "panel_mean_p": float((p * pi).sum()), "share_p_above_0.5": float(pi[p > 0.5].sum())})
    # 5. method interaction I_s = Delta(filtered, s) - Delta(GRPO, s)
    PF = {s: P[("filtered", s)] for s in SEEDS}
    PG = {s: P[("grpo", s)] for s in SEEDS}
    res["method_interaction"] = Y.diff_of(PF, {s: R[("filtered", s)] for s in SEEDS}, PG,
                                          {s: R[("grpo", s)] for s in SEEDS}, lambda k: k, idx)
    # 6. failure decomposition at cap 256 (K = 1)
    dec = []
    res["decomposition_K1"] = {}
    for code in Y.FAIL_CODES:
        summ, Pc, Rc = Y.init_contrast(ev, pairs, w, idx, unit, K=1, code=code)
        res["decomposition_K1"][code] = summ
        for k, v in Pc.items():
            dec.append({"code": code, "method": k[0], "seed": k[1], "D": v})
        for i in INITS:                       # per-init excess of the code rate, by method
            for m in METHODS:
                for s in SEEDS:
                    use = Y.usable(ev, [it("light", s), it("extended", s)])
                    e, _ = Y.branch(ev, it(i, s), ft(i, m, s), w, use, None, K=1, code=code)
                    dec.append({"code": code, "init": i, "method": m, "seed": s, "E_code": e})
    # 7. structure and strategy (exploratory)
    res["structure"] = Y.structure_analysis(ev, probs["xc3"], forms, [
        {"init": i, "method": m, "seed": s, "ti": it(i, s), "tf": ft(i, m, s)}
        for i in INITS for m in METHODS for s in SEEDS])
    json.dump(res, open(os.path.join(a.out, "y1_audit.json"), "w"), indent=1, default=float)
    write_csv(os.path.join(a.out, "y1_pairs.csv"), pair_rows)
    write_csv(os.path.join(a.out, "y1_whole_panel.csv"), wp)
    write_csv(os.path.join(a.out, "y1_decomp.csv"), dec)
    write_csv(os.path.join(a.out, "y1_npmle.csv"), npm)
    print(json.dumps({k: res[k] for k in ("a5_reproduction",)}, default=float))
    print("[y1] done", flush=True)


if __name__ == "__main__":
    main()
