"""A6 look 1 on the confirmation panels yf3 / yf5 (protocol/AMENDMENT-A6-2026-10-26.md Sections 5.2, 6, 7, 8, 10).
Runs ONCE when every wave-1 adapter has its yf evaluation; refuses an incomplete set.

Usage: python scripts/y_look1.py --evals $PROJ_DATA/evals/ext/a6 --matching y4_matching.json --runs $PROJ_DATA/runs
       --out DIR
Writes look1.json (every pre-registered test with its classification), look1_branches.csv (E per branch, both caps),
look1_manip.json (manipulation checks).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from bc2rl import yanalysis as Y  # noqa: E402
from bc2rl.xanalysis import boot_idx, loss_k  # noqa: E402

SEEDS = (11, 22, 33, 44, 55, 66)
METHODS = ("grpo", "filtered")
STEPS = (16, 32, 64, 128)
Y3_ARMS = ("M", "MASK", "T3", "T3E")


def it(n, s):
    return f"b{n}-s{s}"


def ft(n, m, s, arm="M"):
    return f"b{n}-{m}-{arm}-s{s}"


def required():
    inits = [it(n, s) for n in STEPS for s in SEEDS]
    fin = [ft(n, m, s) for n in (16, 128) for m in METHODS for s in SEEDS]
    fin += [ft(n, "grpo", s, arm) for n in (16, 128) for arm in ("MASK", "T3", "T3E") for s in SEEDS]
    fin += [ft(n, "grpo", s) for n in (32, 64) for s in SEEDS]
    return inits, fin


def classify_y2a(d1024, diff, d256):
    lo = [d1024[k][0] for k in ("boot95", "t95")]
    if min(lo) > 0 and d1024["point"] >= 0.5 * d256["point"]:
        return "not explained by truncation"
    if not d1024["both_exclude_0"] and diff["both_exclude_0"]:
        return "largely explained by truncation"
    return "partly explained by truncation"


def classify_shape(inc, total):
    """Section 8.3 shape rule on the three increments (summaries) and E128 - E16 (summary)."""
    pts = [i["point"] for i in inc]
    pos = [min(i["boot95"][0], i["t95"][0]) > 0 for i in inc]
    neg = [max(i["boot95"][1], i["t95"][1]) < 0 for i in inc]
    if any(neg):
        return "non-monotonic"
    if all(p > 0 for p in pts) and sum(pos) >= 2:
        return "gradual"
    for j in range(3):
        others = [k for k in range(3) if k != j]
        if pos[j] and total["point"] > 0 and pts[j] >= 0.7 * total["point"] and \
                not any(inc[k]["both_exclude_0"] for k in others):
            return "threshold"
    return "undetermined"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", required=True)
    ap.add_argument("--matching", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    inits, fins = required()
    c3, c5 = Y.load_summaries(a.evals, "yf3"), Y.load_summaries(a.evals, "yf5")
    missing = [t for t in inits + fins if t not in c3 or t not in c5]
    if missing:
        raise SystemExit(f"yf incomplete, missing {missing}; look 1 refused")
    ev = Y.Evals({t: c3[t] for t in inits + fins})
    ev5 = Y.Evals({t: c5[t] for t in inits + fins})
    idx = boot_idx(len(ev.pids), a.n_boot, 61)
    w = Y.pooled_weights(ev, inits)
    unit = lambda k: k[1]
    res = {"weights_pooled24": w.tolist(), "n_problems": {"yf3": len(ev.pids), "yf5": len(ev5.pids)},
           "n_boot": a.n_boot}
    m_pairs = {(m, s): (it(16, s), ft(16, m, s), it(128, s), ft(128, m, s)) for m in METHODS for s in SEEDS}
    # ---- Y2-a truncation
    D = {}
    for cap in (256, 1024):
        summ, P, R = Y.init_contrast(ev, m_pairs, w, idx, unit, cap=cap)
        res[f"Y2a_delta_cap{cap}"] = summ
        D[cap] = (P, R)
    res["Y2a_delta_256_minus_1024"] = Y.diff_of(*D[256], *D[1024], unit, idx)
    res["Y2a_class"] = classify_y2a(res["Y2a_delta_cap1024"], res["Y2a_delta_256_minus_1024"],
                                    res["Y2a_delta_cap256"])
    # ---- Y2-b failure codes, K = 1
    for cap in (256, 1024):
        tot = Y.init_contrast(ev, m_pairs, w, idx, unit, K=1, cap=cap)[0]
        dec = {}
        for code in Y.FAIL_CODES:
            s_ = Y.init_contrast(ev, m_pairs, w, idx, unit, K=1, cap=cap, code=code)[0]
            s_["share_of_delta_K1"] = s_["point"] / tot["point"] if tot["point"] else None
            s_["accounts_for_part"] = bool(s_["both_exclude_0"] and s_["point"] > 0)
            s_["works_against"] = bool(s_["both_exclude_0"] and s_["point"] < 0)
            dec[code] = s_
        res[f"Y2b_cap{cap}"] = {"delta_K1": tot, "codes": dec}
    # ---- method interaction (Section 7)
    _, P, R = Y.init_contrast(ev, m_pairs, w, idx, unit)
    res["method_interaction"] = Y.diff_of({s: P[("filtered", s)] for s in SEEDS}, {s: R[("filtered", s)] for s in SEEDS},
                                          {s: P[("grpo", s)] for s in SEEDS}, {s: R[("grpo", s)] for s in SEEDS},
                                          lambda k: k, idx)
    res["method_interaction"]["claim"] = ("the objective changes the init difference"
                                          if res["method_interaction"]["both_exclude_0"] else "no method difference shown")
    # ---- Y3 (GRPO; arms M, MASK, T3, T3E)
    rows = []
    E = {}
    for cap in (256, 1024):
        for K in (16, 1, 4, 24):
            if cap == 1024 and K != 16:
                continue
            for s in SEEDS:
                use = Y.usable(ev, [it(16, s), it(128, s)])
                for n in (16, 128):
                    for arm in Y3_ARMS:
                        pt, rp = Y.branch(ev, it(n, s), ft(n, "grpo", s, arm), w, use, idx, K=K, cap=cap)
                        E[(cap, K, n, arm, s)] = (pt, rp)
                        rows.append({"cap": cap, "K": K, "init_step": n, "arm": arm, "seed": s, "E": pt})
    y3 = {}
    for cap, K in ((256, 16), (1024, 16), (256, 1), (256, 4), (256, 24)):
        def ed(arm):                                     # Delta_arm per seed: E(128) - E(16)
            return ({s: E[(cap, K, 128, arm, s)][0] - E[(cap, K, 16, arm, s)][0] for s in SEEDS},
                    {s: E[(cap, K, 128, arm, s)][1] - E[(cap, K, 16, arm, s)][1] for s in SEEDS})
        dM, dMASK, dT3, dT3E = ed("M"), ed("MASK"), ed("T3"), ed("T3E")
        out = {"Delta_M": Y.summarize_contrast(*dM), "C2_Delta_MASK": Y.summarize_contrast(*dMASK),
               "Delta_T3": Y.summarize_contrast(*dT3), "Delta_T3E": Y.summarize_contrast(*dT3E),
               "C1_DeltaM_minus_DeltaMASK": Y.diff_of(*dM, *dMASK, lambda k: k, idx),
               "DeltaMASK_minus_DeltaT3E": Y.diff_of(*dMASK, *dT3E, lambda k: k, idx)}
        for n in (16, 128):
            def eo(arm):
                return ({s: E[(cap, K, n, arm, s)][0] for s in SEEDS}, {s: E[(cap, K, n, arm, s)][1] for s in SEEDS})
            for arm in Y3_ARMS:
                out[f"E_step{n}_{arm}"] = Y.summarize_contrast(*eo(arm))
            out[f"C3_step{n}_M_minus_MASK"] = Y.diff_of(*eo("M"), *eo("MASK"), lambda k: k, idx)
            out[f"C4_step{n}_T3_minus_MASK"] = Y.diff_of(*eo("T3"), *eo("MASK"), lambda k: k, idx)
            out[f"step{n}_MASK_minus_T3E"] = Y.diff_of(*eo("MASK"), *eo("T3E"), lambda k: k, idx)
        y3[f"cap{cap}_K{K}"] = out
    p = y3["cap256_K16"]
    p["interpretation"] = {
        "harder-problem updates contribute to the init difference": p["C1_DeltaM_minus_DeltaMASK"]["sign"] == "+",
        "the init difference arises without harder-problem updates": p["C2_Delta_MASK"]["sign"] == "+"}
    res["Y3"] = y3
    # manipulation checks: yf5 pass@1 change (final - initial) per arm; gradient-carrying completions by input count
    man = {"yf5_pass1_change": {}, "used_by_inputs": {}}
    for n in (16, 128):
        for arm in Y3_ARMS:
            v = [float(ev5.arr(ft(n, "grpo", s, arm), "c").mean() / 32 - ev5.arr(it(n, s), "c").mean() / 48)
                 for s in SEEDS]
            man["yf5_pass1_change"][f"step{n}_{arm}"] = {"seeds": v, "mean": float(np.mean(v))}
    for n, init in ((16, "light"), (128, "extended")):
        for arm in ("mask", "t3"):
            for s in SEEDS:
                f = os.path.join(a.runs, f"bc2rl-y3-{init}-grpo-{arm}-s{s}", "metrics.jsonl")
                if os.path.exists(f):
                    tot = {"3": 0, "5": 0, "10": 0}
                    for line in open(f):
                        r = json.loads(line)
                        for h, v in r.get("used_by_inputs", {}).items():
                            tot[h] += v
                    man["used_by_inputs"][f"step{n}-{arm}-s{s}"] = tot
    man["displacement_norms"] = displacement_norms(a.runs)
    # ---- Y4 dose-response (GRPO, arm M)
    Ed = {}
    for s in SEEDS:
        use = Y.usable(ev, [it(n, s) for n in STEPS])
        for n in STEPS:
            Ed[(n, s)] = Y.branch(ev, it(n, s), ft(n, "grpo", s), w, use, idx)
            rows.append({"cap": 256, "K": 16, "init_step": n, "arm": "M-dose", "seed": s, "E": Ed[(n, s)][0],
                         "bins_used": len(use)})

    def es(n):
        return {s: Ed[(n, s)][0] for s in SEEDS}, {s: Ed[(n, s)][1] for s in SEEDS}
    y4 = {f"E_step{n}": Y.summarize_contrast(*es(n)) for n in STEPS}
    inc = [Y.diff_of(*es(b), *es(a_), lambda k: k, idx) for a_, b in ((16, 32), (32, 64), (64, 128))]
    y4["increments"] = {"I1_32_minus_16": inc[0], "I2_64_minus_32": inc[1], "I3_128_minus_64": inc[2]}
    y4["E128_minus_E16"] = Y.diff_of(*es(128), *es(16), lambda k: k, idx)
    y4["shape"] = classify_shape(inc, y4["E128_minus_E16"])
    mt = json.load(open(a.matching))
    if not mt["matched_pairs"]:
        y4["matched"] = {"status": "overlap inadequate (no pair met Section 8.2)"}
    else:
        P, R = {}, {}
        for mp in mt["matched_pairs"]:
            lo, hi = mp["low"], mp["high"]
            (nl, sl), (nh, sh) = [(int(t.split("-")[0][1:]), int(t.split("-")[1][1:])) for t in (lo, hi)]
            use = Y.usable(ev, [lo, hi])
            a_, ar = Y.branch(ev, lo, ft(nl, "grpo", sl), w, use, idx)
            b_, br = Y.branch(ev, hi, ft(nh, "grpo", sh), w, use, idx)
            P[(lo, hi)], R[(lo, hi)] = b_ - a_, br - ar
        mean = float(np.mean(list(P.values())))
        from bc2rl.xanalysis import ci
        bo = ci(np.mean(np.stack(list(R.values())), axis=0))
        allpos = all(v > 0 for v in P.values())
        y4["matched"] = {"pairs": {f"{k[0]}|{k[1]}": v for k, v in P.items()}, "mean_D": mean, "boot95": list(bo),
                         "all_positive": allpos,
                         "claim": ("BC exposure predicts loss beyond the measured initial distribution"
                                   if bo[0] > 0 and allpos else
                                   "matched initial distribution explains the step effect for these pairs"
                                   if -0.05 <= bo[0] and bo[1] <= 0.05 else "undetermined"),
                         "note": "no seed-level interval: pairs can share adapters"}
    # secondary: regression of E on log2(step) and yd3 pass@1 over the 24 checkpoints
    prof = mt["profiles"]
    X = np.array([[1.0, np.log2(n), prof[it(n, s)]["pass1_3"]] for n in STEPS for s in SEEDS])
    yv = np.array([Ed[(n, s)][0] for n in STEPS for s in SEEDS])
    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    y4["regression"] = {"coef_log2step": float(beta[1]), "coef_pass1_yd3": float(beta[2]),
                        "corr_predictors": float(np.corrcoef(X[:, 1], X[:, 2])[0, 1]),
                        "descriptive_only": bool(abs(np.corrcoef(X[:, 1], X[:, 2])[0, 1]) > 0.8)}
    res["Y4"] = y4
    # whole-panel pass@1 / pass@16 for every adapter (descriptive)
    wp = {}
    for t in inits + fins:
        for pname, e in (("yf3", ev), ("yf5", ev5)):
            n = e.n(t)
            wp[f"{t}|{pname}"] = {"pass1": float(e.arr(t, "c").mean() / n),
                                  "pass16": float((1 - loss_k(n, e.arr(t, "c"), 16)).mean()),
                                  "pass1_x": float(e.arr(t, "c_x").mean() / n)}
    res["whole_panel"] = wp
    json.dump(res, open(os.path.join(a.out, "look1.json"), "w"), indent=1, default=float)
    json.dump(man, open(os.path.join(a.out, "look1_manip.json"), "w"), indent=1, default=float)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(a.out, "look1_branches.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print(f"[look1] Y2a {res['Y2a_class']}; Y3 C1 {p['C1_DeltaM_minus_DeltaMASK']['point']:+.4f} "
          f"C2 {p['C2_Delta_MASK']['point']:+.4f}; Y4 shape {y4['shape']}", flush=True)


def displacement_norms(runs):
    """L2 norm of (adapter - initial adapter) at rounds 64, 128, 256 for the Y3 arms and the M branches (fp32)."""
    import torch
    from safetensors.torch import load_file

    def flat(path):
        sd = load_file(path)
        return torch.cat([sd[k].float().ravel() for k in sorted(sd)])
    out = {}
    for n, init in ((16, "light"), (128, "extended")):
        for s in SEEDS:
            base = flat(os.path.join(runs, f"bc2rl-s1-bc-broad-s{s}", "adapters", f"step{n:04d}.safetensors"))
            for arm, rid in (("M", f"bc2rl-s2-{init}-grpo-s{s}"), ("MASK", f"bc2rl-y3-{init}-grpo-mask-s{s}"),
                             ("T3", f"bc2rl-y3-{init}-grpo-t3-s{s}")):
                for r in (64, 128, 256):
                    f = os.path.join(runs, rid, "adapters", f"round{r:04d}.safetensors")
                    if os.path.exists(f):
                        out[f"step{n}-{arm}-s{s}-r{r}"] = float((flat(f) - base).norm())
    return out


if __name__ == "__main__":
    main()
