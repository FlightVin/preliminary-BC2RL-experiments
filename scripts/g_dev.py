"""A6 Section 12.5 / 12.8 stage-G development analysis (dev seed 11, yd panels; exploratory; CPU).

Inputs: yd summaries of the initial adapters b16-s11 / b128-s11 (48 attempts, split 24) and of the 12 dev finals
g-b<step>-<ARM>-s11 (32 attempts); the run metrics (runs/bc2rl-g-<init>-<arm>-s11/metrics.jsonl, summary.json);
g_validate.json (precision decision and noise floor).
Outputs g_dev.json:
- E (bin-standardized excess loss_K, K = 1, 4, 16; caps 256 and 1,024) per arm and init, with the contrasts of
  Section 12.2 / 12.8 and their extended - light interactions (problem bootstrap only: one seed).
- pass@k and pass^k (k = 1, 4, 16; caps 256 and 1,024), mean output tokens, truncation and rescue per final.
- training diagnostics per run, gate D (D1 validity, D2 exercise) per K = 4 arm, the T4F confirmation trigger.
No outcome enters gate D; the trigger uses yd only.
"""
import argparse
import json
import math
import os

import numpy as np

from bc2rl import yanalysis as Y
from bc2rl.xanalysis import boot_idx

SEED = 11
ARMS = ("C1", "L1", "L4", "S4", "T4", "T4F")
STEP = {"light": 16, "extended": 128}
D2_MIN = 3e-4
CLIP_MIN = 0.01


def it(init):
    return f"b{STEP[init]}-s{SEED}"


def ft(init, arm):
    return f"g-b{STEP[init]}-{arm}-s{SEED}"


def rid(init, arm):
    return f"bc2rl-g-{init}-{arm.lower()}-s{SEED}"


def comb(n, k):
    return math.comb(int(n), int(k)) if 0 <= k <= n else 0


def pass_rates(n, c, k):
    """Mean over problems of unbiased pass@k = 1 - C(n-c,k)/C(n,k) and pass^k = C(c,k)/C(n,k)."""
    d = comb(n, k)
    at = np.mean([1 - comb(n - x, k) / d for x in c])
    hat = np.mean([comb(x, k) / d for x in c])
    return float(at), float(hat)


def ci(reps):
    return [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]


def med(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.median(v)) if v else float("nan")


def run_diag(runs_dir, init, arm):
    rd = os.path.join(runs_dir, rid(init, arm))
    out = {"run": rid(init, arm), "done": os.path.exists(os.path.join(rd, "DONE.json"))}
    mp = os.path.join(rd, "metrics.jsonl")
    if not os.path.exists(mp):
        return out
    rows = [json.loads(l) for l in open(mp)]
    tr = [r for r in rows if not r["skipped"]]
    out.update(rounds=len(rows), trained_rounds=len(tr), opt_steps=rows[-1]["opt_steps"] if rows else 0,
               finite_loss=all(math.isfinite(r["loss"]) for r in tr),
               mean_len=float(np.mean([r["mean_len"] for r in rows])),
               trunc_rate=float(np.mean([r["truncated"] for r in rows])) / 128,
               reward_mean=float(np.mean([r["reward_mean"] for r in rows])),
               dist_from_init_final=rows[-1].get("dist_from_init"),
               kl=[{"round": r["round"], "k1_tok": r["kl_k1_tok"], "k3_tok": r["kl_k3_tok"], "seq": r["kl_seq"]}
                   for r in rows if r.get("kl_seq") is not None])
    sp = os.path.join(rd, "summary.json")
    if os.path.exists(sp):
        s = json.load(open(sp))
        out.update(gpu_s=s.get("gpu_s_total"), counters=s.get("counters"), kl_final=s.get("kl_final"))
    K = max((len(r.get("steps") or []) for r in tr), default=0)
    out["K"] = K
    per_step = {}
    for k in range(K):
        st = [r["steps"][k] for r in tr if len(r.get("steps") or []) > k]
        g = lambda key, j=0: med([(s.get(key) or [None] * 4)[j] for s in st])
        per_step[str(k + 1)] = {
            "grad_norm": med([s["grad_norm"] for s in st]), "update_norm": med([s["update_norm"] for s in st]),
            "tok_abslogr_p50": g("tok_abslogr", 0), "tok_abslogr_p90": g("tok_abslogr", 1),
            "seq_abslogs_p50": g("seq_abslogs", 0), "seq_abslogs_p90": g("seq_abslogs", 1),
            "cum_abslogr_p50": g("cum_abslogr", 0), "cum_abslogr_p90": g("cum_abslogr", 1),
            "frac_outside_mean": float(np.mean([s.get("frac_outside", 0.0) for s in st])) if st else None,
            "frac_clipped_mean": float(np.mean([s.get("frac_clipped", 0.0) for s in st])) if st else None,
            "frac_tokens_clipped_mean": float(np.mean([s.get("frac_tokens_clipped", 0.0) for s in st])) if st else None}
        if k == K - 1 and st and st[0].get("by_pos"):
            per_step[str(k + 1)]["by_pos"] = {b: {"cum_abs_p50": med([s["by_pos"][b]["cum_abs"][0] for s in st]),
                                                  "frac_clipped": med([s["by_pos"][b].get("frac_clipped") for s in st])}
                                              for b in st[0]["by_pos"]}
    out["per_step"] = per_step
    if K > 1:
        out["step1_ratio_exactly_one"] = all(r["steps"][0]["tok_abslogr"][-1] == 0.0 and
                                             r["steps"][0]["seq_abslogs"][-1] == 0.0 for r in tr)
        out["clip_frac_steps2to4"] = float(np.mean([np.mean([r["steps"][k]["frac_clipped"] for k in range(1, K)])
                                                    for r in tr]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--validate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {"seed": SEED, "panel": "yd3", "note": "exploratory dev analysis (one seed; problem bootstrap only)"}
    val = json.load(open(a.validate))
    res["precision"] = val["precision_decision"]
    b1 = [r["V4_steps"][0] for e in val["adapters"] for r in e["batches"] if "V4_steps" in r]
    floor = {u: (max(s[f"med_abs_noise_{u}"] for s in b1) if res["precision"] == "bf16" else 0.0)
             for u in ("seq", "tok", "cum")}
    res["noise_floor"] = floor
    # training diagnostics and gate D (no outcome used)
    diag = {f"{i}-{arm}": run_diag(a.runs, i, arm) for i in STEP for arm in ARMS}
    res["runs"] = diag
    d1 = all(d.get("done") and d.get("rounds") == 256 and d.get("finite_loss") for d in diag.values()) and \
        all(d.get("step1_ratio_exactly_one", True) for d in diag.values())
    gate = {"D1_valid": bool(d1)}
    for arm, key, unit in (("L4", "tok_abslogr_p50", "tok"), ("S4", "seq_abslogs_p50", "seq"),
                           ("T4", "cum_abslogr_p50", "cum")):
        vals = [diag[f"{i}-{arm}"].get("per_step", {}).get("4", {}).get(key) for i in STEP]
        clips = [diag[f"{i}-{arm}"].get("clip_frac_steps2to4") for i in STEP]
        ok = all(v is not None and v >= D2_MIN and v >= 3 * floor[unit] for v in vals) and \
            all(c is not None and c >= CLIP_MIN for c in clips)
        gate[f"D2_{arm}"] = {"pass": bool(ok), "step4_median": vals, "clip_frac_steps2to4": clips,
                             "thresholds": {"median_min": max(D2_MIN, 3 * floor[unit]), "clip_min": CLIP_MIN}}
    res["gate_D"] = gate
    # outcomes on yd3 (exploratory)
    c3, c5 = Y.load_summaries(a.evals, "yd3"), Y.load_summaries(a.evals, "yd5")
    inits = [it(i) for i in STEP]
    finals = [ft(i, arm) for i in STEP for arm in ARMS]
    hist = [f"b{STEP[i]}-grpo-M-s{SEED}" for i in STEP]
    missing = [t for t in inits + finals if t not in c3]
    if missing:
        raise SystemExit(f"yd3 incomplete, missing {missing}")
    tags = inits + finals + [t for t in hist if t in c3]
    ev = Y.Evals({t: c3[t] for t in tags})
    w = Y.pooled_weights(ev, inits)
    use = Y.usable(ev, inits)
    idx = boot_idx(len(ev.pids), a.n_boot, 7)
    res.update(weights=w.tolist(), usable_bins=use, n_problems=len(ev.pids))
    E = {}
    for K in (1, 4, 16):
        for cap in (256, 1024):
            for i in STEP:
                for arm in ARMS + ("M",):
                    f = ft(i, arm) if arm != "M" else f"b{STEP[i]}-grpo-M-s{SEED}"
                    if ev.has(f):
                        E[(K, cap, i, arm)] = Y.branch(ev, it(i), f, w, use, idx, K=K, cap=cap)
    out_E = {}
    for (K, cap, i, arm), (pt, reps) in E.items():
        out_E[f"K{K}_cap{cap}_{i}_{arm}"] = {"point": float(pt), "ci95": ci(reps)}
    res["E"] = out_E
    contrasts = {"norm L1-C1": ("L1", "C1"), "reuse L4-L1": ("L4", "L1"), "seq S4-L4": ("S4", "L4"),
                 "ctpo_vs_tok T4-L4": ("T4", "L4"), "ctpo_vs_seq T4-S4": ("T4", "S4"),
                 "schedule T4-T4F": ("T4", "T4F"), "C1-hist M": ("C1", "M")}
    con = {}
    for K in (1, 4, 16):
        for cap in (256, 1024):
            for name, (x, y) in contrasts.items():
                if (K, cap, "light", y) not in E or (K, cap, "light", x) not in E:
                    continue
                d = {i: (E[(K, cap, i, x)][0] - E[(K, cap, i, y)][0], E[(K, cap, i, x)][1] - E[(K, cap, i, y)][1])
                     for i in STEP}
                mean_pt = (d["light"][0] + d["extended"][0]) / 2
                mean_r = (d["light"][1] + d["extended"][1]) / 2
                int_pt = d["extended"][0] - d["light"][0]
                int_r = d["extended"][1] - d["light"][1]
                con[f"K{K}_cap{cap}_{name}"] = {
                    "light": {"point": float(d["light"][0]), "ci95": ci(d["light"][1])},
                    "extended": {"point": float(d["extended"][0]), "ci95": ci(d["extended"][1])},
                    "mean": {"point": float(mean_pt), "ci95": ci(mean_r)},
                    "interaction_ext_minus_light": {"point": float(int_pt), "ci95": ci(int_r)}}
    res["contrasts"] = con
    k16 = lambda arm: (E[(16, 256, "light", arm)][0] + E[(16, 256, "extended", arm)][0]) / 2
    res["T4F_trigger"] = {"E_T4_minus_L4": float(k16("T4") - k16("L4")), "E_T4_minus_S4": float(k16("T4") - k16("S4")),
                          "triggered": bool(k16("T4") - k16("L4") < 0 or k16("T4") - k16("S4") < 0)}
    # pass@k / pass^k, length, truncation per final (and the initial adapters, baseline half)
    rates = {}
    for t in finals + inits:
        n = ev.n(t)
        r = {}
        for cap, key in ((256, "c"), (1024, "c_x")):
            c = ev.arr(t, key) if t in finals else ev.arr(t, "c_b" + ("" if cap == 256 else "_x"))
            nn = n if t in finals else n - n // 2
            for k in (1, 4, 16):
                r[f"pass@{k}_cap{cap}"], r[f"pass^{k}_cap{cap}"] = pass_rates(nn, c, k)
        s = json.load(open(os.path.join(a.evals, f"{t}.yd3.summary.json")))
        e3 = s.get("ext", {}).get("3", {})
        r.update(mean_tokens_256=float(np.mean([float(x["mean_tokens"]) for x in s["counts"]])),
                 mean_tokens_x=e3.get("mean_tokens_x"), trunc_256=e3.get("trunc_256"), trunc_x=e3.get("trunc_x"),
                 rescued=e3.get("rescued"))
        rates[t] = r
    res["rates_yd3"] = rates
    json.dump(res, open(os.path.join(a.out, "g_dev.json"), "w"), indent=1)
    print(f"[g-dev] precision {res['precision']}; gate D {json.dumps(gate)}; T4F trigger {res['T4F_trigger']}")


if __name__ == "__main__":
    main()
