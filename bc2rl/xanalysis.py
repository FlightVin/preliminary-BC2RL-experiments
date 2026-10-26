"""Pre-registered confirmation analysis X5 (protocol/AMENDMENT-A5-2026-10-26.md, Section 3). Written and tested on
synthetic data before any extension outcome exists.

Inputs per problem x on a fresh panel:
  initial adapter (init, seed): 48 attempts; c_a = successes in attempts 0-23 (selection), c_b = 24-47 (baseline)
  final adapter (init, method, seed, round 256): 32 fresh attempts; c_f
Estimators (unbiased for a fixed success probability p and independent attempts):
  loss_k(n, c) = C(n - c, k) / C(n, k)      for (1 - p)^k     (= 1 - pass@k)
  rel_k(n, c)  = C(c, k) / C(n, k)          for p^k           (= pass^k)
Excess loss of x at K: e = loss_K(32, c_f) - loss_K(24, c_b). Selection uses c_a only, so E[e | selected] is the true
change in (1 - p)^K on the selected set (no regression to the mean).
Primary: bins of c_a (B1 1-2, B2 3-5, B3 6-11 of 24), weights fixed from pooled selection counts, branch statistic
E = sum_b w_b mean_b(e); Delta(m, s) = E(extended, m, s) - E(light, m, s); primary = mean of the 6 Deltas, K = 16.
"""
from __future__ import annotations

import glob
import json
import os
from math import comb

import numpy as np

INITS = ("light", "extended")
METHODS = ("grpo", "filtered")
SEEDS = (11, 22, 33)
N_SEL = N_BASE = 24
N_FIN = 32
PRIMARY_K = 16
PRIMARY_BINS = ((1, 2), (3, 5), (6, 11))
HIGH_BIN = ((12, 23),)
MIN_BIN_N = 10


def loss_k(n, c, k):
    c = np.asarray(c)
    den = comb(n, k)
    return np.array([comb(n - int(x), k) / den for x in c.ravel()]).reshape(c.shape)


def rel_k(n, c, k):
    c = np.asarray(c)
    den = comb(n, k)
    return np.array([comb(int(x), k) / den for x in c.ravel()]).reshape(c.shape)


def bin_of(c_sel, bins):
    """Index of the bin containing each selection count, -1 outside every bin."""
    out = np.full(len(c_sel), -1)
    for j, (lo, hi) in enumerate(bins):
        out[(c_sel >= lo) & (c_sel <= hi)] = j
    return out


def bin_weights(sel_by_init_seed, bins):
    """Pooled share of each bin over every initial adapter (selection counts only; fixed before outcomes)."""
    tot = np.zeros(len(bins))
    for c_sel in sel_by_init_seed.values():
        b = bin_of(np.asarray(c_sel), bins)
        tot += np.array([(b == j).sum() for j in range(len(bins))])
    return tot / tot.sum()


def usable_bins(b_light, b_ext, nbins, min_n=MIN_BIN_N):
    """Bins with >= min_n selected problems for both initial adapters (decided from selection counts only)."""
    return [j for j in range(nbins) if (b_light == j).sum() >= min_n and (b_ext == j).sum() >= min_n]


def standardized(vals, b, w, use, idx=None):
    """sum_{j in use} w_j mean(vals over problems in bin j) / sum_{j in use} w_j; idx = resampled problem indices
    (rows of a 2-D array give one bootstrap replicate each)."""
    if idx is None:
        idx = np.arange(len(vals))[None, :]
    v, bb = vals[idx], b[idx]
    num = np.zeros(idx.shape[0])
    wsum = np.zeros(idx.shape[0])
    for j in use:
        m = bb == j
        cnt = m.sum(1)
        ok = cnt > 0
        mean = np.where(ok, (v * m).sum(1) / np.maximum(cnt, 1), 0.0)
        num += np.where(ok, w[j] * mean, 0.0)
        wsum += np.where(ok, w[j], 0.0)
    return num / wsum


class Panel:
    """Aligned count arrays for one fresh panel. counts: {tag: {pid: row}} with keys n, c, c_a, c_b."""

    def __init__(self, counts, pids=None):
        self.pids = sorted(next(iter(counts.values()))) if pids is None else pids
        for tag, rows in counts.items():
            if sorted(rows) != self.pids:
                raise ValueError(f"{tag}: problem set differs from the panel")
        self.c = {t: np.array([int(r[p]["c"]) for p in self.pids]) for t, r in counts.items()}
        self.n = {t: {int(r[p]["n"]) for p in self.pids} for t, r in counts.items()}
        self.ca = {t: np.array([int(r[p]["c_a"]) for p in self.pids]) for t, r in counts.items()
                   if str(r[self.pids[0]].get("c_a", "")) != ""}
        self.cb = {t: np.array([int(r[p]["c_b"]) for p in self.pids]) for t, r in counts.items()
                   if str(r[self.pids[0]].get("c_b", "")) != ""}


def init_tag(init, seed):
    return f"{init}-s{seed}"


def final_tag(init, method, seed):
    return f"{init}-{method}-s{seed}-r256"


def comparison(panel, K=PRIMARY_K, bins=PRIMARY_BINS, swap=False, measure="loss", idx=None, use_by_pair=None):
    """Bin-standardized excess (final - baseline) per branch and Delta(extended - light) per (method, seed).

    measure: "loss" -> loss_K; "rel" -> rel_K (consistency). swap: select on c_b, baseline from c_a.
    Returns dict with per-branch E, per-pair Delta, the mean Delta, used bins and weights (all arrays if idx given).
    """
    sel = {(i, s): (panel.cb if swap else panel.ca)[init_tag(i, s)] for i in INITS for s in SEEDS}
    base = {(i, s): (panel.ca if swap else panel.cb)[init_tag(i, s)] for i in INITS for s in SEEDS}
    f = loss_k if measure == "loss" else rel_k
    w = bin_weights(sel, bins)
    E, D, used = {}, {}, {}
    for m in METHODS:
        for s in SEEDS:
            b = {i: bin_of(sel[(i, s)], bins) for i in INITS}
            use = use_by_pair[(m, s)] if use_by_pair is not None else usable_bins(b["light"], b["extended"], len(bins))
            used[(m, s)] = use
            if not use:
                D[(m, s)] = np.nan
                continue
            for i in INITS:
                e = f(N_FIN, panel.c[final_tag(i, m, s)], K) - f(N_BASE, base[(i, s)], K)
                E[(i, m, s)] = standardized(e, b[i], w, use, idx)
            D[(m, s)] = E[("extended", m, s)] - E[("light", m, s)]
    vals = [D[k] for k in D if not np.all(np.isnan(D[k]))]
    return {"E": E, "D": D, "mean": np.mean(vals, axis=0), "used": used, "w": w}


def boot_idx(n, n_boot=10000, seed=0):
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def ci(samples):
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def decide(point, lo, hi, d_pairs):
    pos = sum(1 for v in d_pairs if v > 0)
    if lo > 0 and pos >= 5:
        return "CONFIRMED"
    if hi < 0:
        return "CONTRADICTED"
    return "NOT CONFIRMED"


def primary(panel, n_boot=10000, seed=0, **kw):
    pt = comparison(panel, **kw)
    idx = boot_idx(len(panel.pids), n_boot, seed)
    bs = _boot_fixed_bins(panel, pt, idx, **kw)      # replicates keep the bins chosen on the observed selection counts
    lo, hi = ci(bs["mean"])
    out = {"K": kw.get("K", PRIMARY_K), "swap": kw.get("swap", False), "measure": kw.get("measure", "loss"),
           "mean_delta": float(pt["mean"]), "ci95": [lo, hi], "weights": pt["w"].tolist(),
           "pairs": {f"{m}-s{s}": {"delta": float(pt["D"][(m, s)]), "ci95": list(ci(bs["D"][(m, s)])),
                                   "bins_used": pt["used"][(m, s)],
                                   "E_light": float(pt["E"][("light", m, s)]),
                                   "E_extended": float(pt["E"][("extended", m, s)])}
                     for m in METHODS for s in SEEDS if pt["used"][(m, s)]}}
    for m in METHODS:
        v = np.mean([bs["D"][(m, s)] for s in SEEDS if pt["used"][(m, s)]], axis=0)
        p = float(np.mean([pt["D"][(m, s)] for s in SEEDS if pt["used"][(m, s)]]))
        out[f"mean_delta_{m}"] = {"point": p, "ci95": list(ci(v))}
    out["decision"] = decide(out["mean_delta"], lo, hi, [v["delta"] for v in out["pairs"].values()])
    out["pairs_positive"] = sum(v["delta"] > 0 for v in out["pairs"].values())
    return out


def _boot_fixed_bins(panel, pt, idx, **kw):
    r = comparison(panel, idx=idx, use_by_pair=pt["used"], **kw)
    D = {k: v for k, v in r["D"].items() if pt["used"][k]}
    return {"D": D, "mean": np.mean(list(D.values()), axis=0)}


def per_bin_table(panel, K=PRIMARY_K, bins=PRIMARY_BINS + HIGH_BIN, swap=False):
    """Descriptive rows: per initial adapter and bin, n, mean selection and baseline rates, and per method the mean
    final loss_K, baseline loss_K and excess. Shows how closely the bins match initial success across inits."""
    rows = []
    for i in INITS:
        for s in SEEDS:
            sel = (panel.cb if swap else panel.ca)[init_tag(i, s)]
            base = (panel.ca if swap else panel.cb)[init_tag(i, s)]
            b = bin_of(sel, bins)
            for j, (lo, hi) in enumerate(bins):
                msk = b == j
                row = {"init": i, "seed": s, "bin": f"{lo}-{hi}", "n": int(msk.sum()),
                       "sel_rate": float(sel[msk].mean() / N_SEL) if msk.any() else None,
                       "base_rate": float(base[msk].mean() / N_BASE) if msk.any() else None}
                for m in METHODS:
                    cf = panel.c[final_tag(i, m, s)]
                    if msk.any():
                        fl = loss_k(N_FIN, cf[msk], K).mean()
                        bl = loss_k(N_BASE, base[msk], K).mean()
                        row.update({f"{m}_final_loss": float(fl), f"{m}_base_loss": float(bl),
                                    f"{m}_excess": float(fl - bl), f"{m}_final_rate": float(cf[msk].mean() / N_FIN)})
                rows.append(row)
    return rows


def curve_family(panel, kmax, measure, idx):
    """Delta(K) curves, K = 1..kmax, mean over the 6 pairs, with a simultaneous band over K (max |error|)."""
    pts, reps = [], []
    for K in range(1, kmax + 1):
        pt = comparison(panel, K=K, measure=measure)
        pts.append(float(pt["mean"]))
        reps.append(_boot_fixed_bins(panel, pt, idx, K=K, measure=measure)["mean"])
    reps = np.array(reps)                       # (kmax, n_boot)
    pts = np.array(pts)
    dev = np.abs(reps - pts[:, None]).max(0)
    c95 = float(np.percentile(dev, 95))
    return [{"K": K + 1, "delta": float(pts[K]), "pw_lo": float(np.percentile(reps[K], 2.5)),
             "pw_hi": float(np.percentile(reps[K], 97.5)), "sim_lo": float(pts[K] - c95),
             "sim_hi": float(pts[K] + c95)} for K in range(kmax)]


def full_panel(panel, kmax_init=48, kmax_fin=N_FIN, idx=None):
    """Whole-panel pass@k / pass^k for every adapter (k up to its attempt count) and paired final - initial
    contrasts for k = 1..32 (initial uses all 48 attempts) with pointwise and simultaneous (over k, per family) bands."""
    rows, contr = [], []
    for t, c in panel.c.items():
        n = max(panel.n[t])
        for k in range(1, n + 1):
            rows.append({"tag": t, "k": k, "pass@": float((1 - loss_k(n, c, k)).mean()),
                         "pass^": float(rel_k(n, c, k).mean())})
    if idx is None:
        return rows, contr
    for i in INITS:
        for m in METHODS:
            for s in SEEDS:
                ti, tf = init_tag(i, s), final_tag(i, m, s)
                for fam, f in (("pass@", lambda n, c, k: 1 - loss_k(n, c, k)), ("pass^", rel_k)):
                    pts, reps = [], []
                    for k in range(1, kmax_fin + 1):
                        d = f(N_FIN, panel.c[tf], k) - f(max(panel.n[ti]), panel.c[ti], k)
                        pts.append(d.mean())
                        reps.append(d[idx].mean(1))
                    reps = np.array(reps)
                    dev = np.abs(reps - np.array(pts)[:, None]).max(0)
                    c95 = float(np.percentile(dev, 95))
                    for k in range(kmax_fin):
                        contr.append({"branch": tf, "family": fam, "k": k + 1, "diff": float(pts[k]),
                                      "pw_lo": float(np.percentile(reps[k], 2.5)),
                                      "pw_hi": float(np.percentile(reps[k], 97.5)),
                                      "sim_lo": float(pts[k] - c95), "sim_hi": float(pts[k] + c95)})
    return rows, contr


def consistent_subgroup(panel, kmax=N_BASE, thresh=22):
    """Problems with c_a >= thresh (of 24): baseline (c_b, n = 24) vs final (n = 32) pass^k, k = 1..24."""
    out = []
    for i in INITS:
        for s in SEEDS:
            msk = panel.ca[init_tag(i, s)] >= thresh
            for m in METHODS:
                row = {"init": i, "method": m, "seed": s, "n": int(msk.sum())}
                if msk.sum() >= 12:
                    cf, cb = panel.c[final_tag(i, m, s)][msk], panel.cb[init_tag(i, s)][msk]
                    row.update({f"rel{k}_base": float(rel_k(N_BASE, cb, k).mean()) for k in (1, 4, 8, 16, 24)})
                    row.update({f"rel{k}_final": float(rel_k(N_FIN, cf, k).mean()) for k in (1, 4, 8, 16, 24)})
                out.append(row)
    return out


def load_counts(summary_dir, panel_name):
    """{tag: {pid: row}} from `<tag>.<panel>.summary.json` files written by `bc2rl xeval`."""
    out = {}
    for f in sorted(glob.glob(os.path.join(summary_dir, f"*.{panel_name}.summary.json"))):
        s = json.load(open(f))
        out[s["tag"]] = {r["pid"]: r for r in s["counts"]}
    return out


def required_tags():
    return [init_tag(i, s) for i in INITS for s in SEEDS] + \
        [final_tag(i, m, s) for i in INITS for m in METHODS for s in SEEDS]
