"""A6 Section 12.4 clip calibration from the four unclipped K = 4 probe runs (CPU).

e_tok = 85th percentile of |r_it - 1| over tokens of rows with A != 0, steps 2-4, all rounds, both initial adapters
        (`grpo_len` probes); e_seq = the same for |s_i - 1| (`gspo` probes). 2 significant digits, symmetric band.
CTPO (Section 12.8): the reference band shape is kept (log-space [-e_lo t^p, e_hi t^p], e_hi = 2 e_lo, dual clip
e_c = e_hi); e_lo is solved so that 15% of the cumulative log-ratios (tokens of rows with A != 0, steps 2-4, both
initial adapters, `ctpo` probes) fall outside, for p = 0.5 (position-adaptive, arm T4) and p = 0 (fixed, arm T4F).
Also reported: the fractions outside the paper bands (token 0.2 / 0.27, sequence 3e-4 / 4e-4) and per-init, per-step
quantiles. Writes g_calibration.json.
"""
import argparse
import glob
import json
import os

import numpy as np

Q = 0.85
PAPER = {"tok": (0.2, 0.27), "seq": (3e-4, 4e-4)}


def sig2(x):
    return float(f"{x:.2g}")


def load(runs, init, obj, unit):
    out = {k: [] for k in (2, 3, 4)}
    files = sorted(glob.glob(os.path.join(runs, f"bc2rl-gp-{init}-{obj}-s11", "ratios", "round*.npz")))
    for f in files:
        z = np.load(f)
        for k in (2, 3, 4):
            if f"{unit}{k}" in z.files:
                out[k].append(z[f"{unit}{k}"])
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}, len(files)


def solve_shape(x, target=1 - Q, ratio=2.0):
    """e_lo with mean((x < -e_lo) | (x > ratio e_lo)) = target (bisection; x float64)."""
    lo, hi = 0.0, max(float(np.abs(x).max()), 1e-12)
    for _ in range(200):
        mid = (lo + hi) / 2
        if np.mean((x < -mid) | (x > ratio * mid)) > target:
            lo = mid
        else:
            hi = mid
    return hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"quantile": Q, "probes": {}}
    pooled = {}
    for obj, unit in (("grpolen", "tok"), ("gspo", "seq")):
        allv = []
        for init in ("light", "extended"):
            d, nf = load(a.runs, init, obj, unit)
            dev = {k: np.abs(np.expm1(v.astype(np.float64))) for k, v in d.items()}
            lo, hi = PAPER[unit]
            res["probes"][f"{init}-{obj}"] = {
                "rounds_with_dump": nf,
                "per_step": {str(k): {"n": int(v.size), **({"p50": float(np.quantile(v, .5)), "p85": float(np.quantile(v, Q)),
                                      "p99": float(np.quantile(v, .99))} if v.size else {})} for k, v in dev.items()},
                "frac_outside_paper_band": float(np.mean(np.concatenate(
                    [(np.expm1(v) < -lo) | (np.expm1(v) > hi) for v in d.values()]))) if any(v.size for v in d.values()) else None}
            allv += [v for v in dev.values() if v.size]
        pooled[unit] = np.concatenate(allv) if allv else np.zeros(0)
    if pooled["tok"].size == 0 or pooled["seq"].size == 0:
        raise SystemExit("no probe ratio dumps found")
    xs = {0.5: [], 0.0: []}
    for init in ("light", "extended"):
        cum, nf = load(a.runs, init, "ctpo", "cum")
        pos, _ = load(a.runs, init, "ctpo", "pos")
        for k in (2, 3, 4):
            if cum[k].size:
                c = cum[k].astype(np.float64)
                xs[0.5].append(c / np.sqrt(np.maximum(pos[k].astype(np.float64), 1.0)))
                xs[0.0].append(c)
        res["probes"][f"{init}-ctpo"] = {"rounds_with_dump": nf, "n": int(sum(v.size for v in cum.values())),
                                        "frac_outside_reference_band": float(np.mean(np.concatenate(
                                            [(x < -0.025) | (x > 0.05) for x in xs[0.5][-3:]]))) if xs[0.5] else None}
    if not xs[0.5]:
        raise SystemExit("no ctpo probe dumps found")
    for pw, name in ((0.5, "ctpo"), (0.0, "ctpo_fixed")):
        x = np.concatenate(xs[pw])
        e = sig2(solve_shape(x))
        res[name] = {"power": pw, "clip_low": e, "clip_high": sig2(2 * e), "clip_c": sig2(2 * e), "n": int(x.size),
                     "frac_outside_at_rounded": float(np.mean((x < -e) | (x > sig2(2 * e))))}
    res["eps_tok_raw"] = float(np.quantile(pooled["tok"], Q))
    res["eps_seq_raw"] = float(np.quantile(pooled["seq"], Q))
    res["eps_tok"], res["eps_seq"] = sig2(res["eps_tok_raw"]), sig2(res["eps_seq_raw"])
    res["n_tok"], res["n_seq"] = int(pooled["tok"].size), int(pooled["seq"].size)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "g_calibration.json"), "w"), indent=1)
    print(f"[g-calibrate] eps_tok={res['eps_tok']} (raw {res['eps_tok_raw']:.4g}, n {res['n_tok']}); "
          f"eps_seq={res['eps_seq']} (raw {res['eps_seq_raw']:.4g}, n {res['n_seq']}); ctpo {res['ctpo']}; "
          f"ctpo_fixed {res['ctpo_fixed']}")


if __name__ == "__main__":
    main()
