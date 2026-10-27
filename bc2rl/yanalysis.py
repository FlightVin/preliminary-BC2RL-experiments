"""Amendment A6 analyses (protocol/AMENDMENT-A6-2026-10-26.md). Written and tested on synthetic data before any A6
outcome exists. The X5 estimator (`xanalysis`) is reused unchanged:

  excess e = loss_K(n_f, c_f) - loss_K(24, c_b); bins of the cap-256 selection count c_a (1-2, 3-5, 6-11 of 24);
  branch statistic E = sum_b w_b mean_b(e) / sum_b w_b over usable bins (>= 10 problems for every initial adapter of
  the comparison).

Additions: outcomes at cap 256 or at the extended cap (fields `*_x`); the failure-code decomposition at K = 1 (the
failure rate is the sum of the code rates, so the decomposition is exact); contrasts with a paired problem bootstrap
and a seed-level Student-t interval; the NPMLE of the distribution of p over problems; the Y4 matching rule.
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter, defaultdict
from math import comb

import numpy as np

from .xanalysis import PRIMARY_BINS, MIN_BIN_N, bin_of, ci, loss_k, rel_k, standardized

CODES = "STFIAPW"          # success, truncated, format, illegal operand, arithmetic, incomplete, wrong target
FAIL_CODES = "TFIAPW"
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 17: 2.110, 19: 2.093, 23: 2.069}


def t975(df):
    return T975.get(df) or T975[max(k for k in T975 if k <= df)]


class Evals:
    """Per-problem rows of one panel: {tag: {pid: row}} (rows from `bc2rl xeval` summaries or built from records)."""

    def __init__(self, counts, tags=None):
        tags = list(tags or sorted(counts))
        self.pids = sorted(counts[tags[0]])
        for t in tags:
            if sorted(counts[t]) != self.pids:
                raise ValueError(f"{t}: problem set differs from the panel")
        self.rows = {t: [counts[t][p] for p in self.pids] for t in tags}
        self._cache = {}

    def has(self, tag):
        return tag in self.rows

    def n(self, tag):
        ns = {int(r["n"]) for r in self.rows[tag]}
        if len(ns) != 1:
            raise ValueError(f"{tag}: unequal attempt counts {ns}")
        return ns.pop()

    def arr(self, tag, key):
        k = (tag, key)
        if k not in self._cache:
            self._cache[k] = np.array([int(r[key]) for r in self.rows[tag]])
        return self._cache[k]

    def codes(self, tag, cap=256):
        """(problems x attempts) array of one-letter failure codes at cap 256 ('codes') or the extended cap."""
        k = (tag, "codes", cap)
        if k not in self._cache:
            key = "codes" if cap == 256 else "x_codes"
            self._cache[k] = np.array([list(r[key]) for r in self.rows[tag]])
        return self._cache[k]


def sfx(cap):
    return "" if cap == 256 else "_x"


def per_problem(ev, init_tag, fin_tag, K=16, cap=256, measure="excess", code=None):
    """Per-problem values for one (initial, final) adapter pair.

    measure: "excess" (final - baseline), "final", "base"; with code=None the scale is loss_K (rel_K if
    measure starts with "rel_"); with code = one letter, the scale is the rate of that code (K = 1 decomposition).
    """
    split = ev.n(init_tag) // 2
    nb = ev.n(init_tag) - split
    nf = ev.n(fin_tag) if fin_tag else None
    if code is not None:
        base = (ev.codes(init_tag, cap)[:, split:] == code).mean(1)
        fin = (ev.codes(fin_tag, cap) == code).mean(1) if fin_tag else None
    else:
        f = rel_k if measure.startswith("rel_") else loss_k
        base = f(nb, ev.arr(init_tag, "c_b" + sfx(cap)), K)
        fin = f(nf, ev.arr(fin_tag, "c" + sfx(cap)), K) if fin_tag else None
    m = measure.replace("rel_", "")
    if m == "base":
        return base
    if m == "final":
        return fin
    return fin - base


def bins_for(ev, init_tag, bins=PRIMARY_BINS):
    return bin_of(ev.arr(init_tag, "c_a"), bins)


def pooled_weights(ev, init_tags, bins=PRIMARY_BINS):
    tot = np.zeros(len(bins))
    for t in init_tags:
        b = bins_for(ev, t, bins)
        tot += np.array([(b == j).sum() for j in range(len(bins))])
    return tot / tot.sum()


def usable(ev, init_tags, bins=PRIMARY_BINS, min_n=MIN_BIN_N):
    bs = [bins_for(ev, t, bins) for t in init_tags]
    return [j for j in range(len(bins)) if all((b == j).sum() >= min_n for b in bs)]


def branch(ev, init_tag, fin_tag, w, use, idx=None, bins=PRIMARY_BINS, **kw):
    """(point, bootstrap replicates or None) of the bin-standardized statistic for one branch."""
    vals = per_problem(ev, init_tag, fin_tag, **kw)
    b = bins_for(ev, init_tag, bins)
    pt = standardized(vals, b, w, use)
    reps = standardized(vals, b, w, use, idx) if idx is not None else None
    return pt, reps


def summarize_contrast(points, reps):
    """points/reps: {unit: value / replicate array}; units are seeds (or pairs grouped by seed beforehand).
    The estimate is the mean over units; boot95 = percentile interval of the mean of replicates; t95 over units."""
    keys = list(points)
    pts = np.array([points[k] for k in keys], dtype=float)
    mean = float(pts.mean())
    out = {"point": mean, "units": {str(k): float(points[k]) for k in keys}, "n_units": len(keys),
           "n_pos": int((pts > 0).sum())}
    if reps is not None:
        r = np.mean(np.stack([reps[k] for k in keys]), axis=0)
        out["boot95"] = list(ci(r))
    if len(keys) >= 2:
        h = t975(len(keys) - 1) * pts.std(ddof=1) / np.sqrt(len(keys))
        out["t95"] = [mean - h, mean + h]
    lo = [out[k][0] for k in ("boot95", "t95") if k in out]
    hi = [out[k][1] for k in ("boot95", "t95") if k in out]
    out["both_exclude_0"] = bool(len(lo) == 2 and (min(lo) > 0 or max(hi) < 0))
    out["sign"] = "+" if lo and min(lo) > 0 and len(lo) == 2 else "-" if hi and max(hi) < 0 and len(hi) == 2 else "0"
    return out


def init_contrast(ev, pairs, w, idx=None, unit_of=None, **kw):
    """Delta per pair = E(high init, final_high) - E(low init, final_low).

    pairs: {pair_key: (lo_init, lo_final, hi_init, hi_final)}. unit_of(pair_key) -> unit (seed) used for the
    seed-level interval; pairs of one unit are averaged first. Returns (summary, per-pair points, per-pair reps)."""
    P, R = {}, {}
    for k, (li, lf, hi_, hf) in pairs.items():
        use = usable(ev, [li, hi_])
        if not use:
            continue
        a, ar = branch(ev, li, lf, w, use, idx, **kw)
        b, br = branch(ev, hi_, hf, w, use, idx, **kw)
        P[k] = b - a
        R[k] = None if idx is None else br - ar
    return group_units(P, R, unit_of, idx), P, R


def group_units(P, R, unit_of, idx):
    unit_of = unit_of or (lambda k: k)
    up, ur = {}, {}
    for k in P:
        up.setdefault(unit_of(k), []).append(P[k])
        if idx is not None:
            ur.setdefault(unit_of(k), []).append(R[k])
    pts = {u: float(np.mean(v)) for u, v in up.items()}
    reps = {u: np.mean(np.stack(v), axis=0) for u, v in ur.items()} if idx is not None else None
    return summarize_contrast(pts, reps)


def diff_of(PA, RA, PB, RB, unit_of, idx):
    """Contrast A - B over the keys present in both (per-pair points and replicates)."""
    keys = [k for k in PA if k in PB]
    P = {k: PA[k] - PB[k] for k in keys}
    R = {k: RA[k] - RB[k] for k in keys} if idx is not None else None
    return group_units(P, R, unit_of, idx)


# ----------------------------------------------------------------------------- distribution of p (NPMLE)

def binom_pmf(n, c, p):
    c = np.asarray(c)[:, None]
    coef = np.array([comb(n, int(x)) for x in c.ravel()])[:, None]
    return coef * p[None, :] ** c * (1 - p[None, :]) ** (n - c)


def npmle(c, n, grid=101, iters=3000):
    """Nonparametric MLE (EM on a fixed grid) of the distribution of p over problems from counts c of n attempts."""
    p = np.linspace(0, 1, grid)
    L = binom_pmf(n, c, p)
    pi = np.full(grid, 1.0 / grid)
    for _ in range(iters):
        post = L * pi
        post /= post.sum(1, keepdims=True)
        pi = post.mean(0)
    return p, pi


def posterior_mean_by_bin(p, pi, n_sel, bins=PRIMARY_BINS):
    """E[p | c_a in bin] under the fitted distribution, for a selection count of n_sel attempts."""
    out = []
    for lo, hi in bins:
        cs = np.arange(lo, hi + 1)
        lik = binom_pmf(n_sel, cs, p) * pi[None, :]
        out.append(float((lik * p).sum() / lik.sum()))
    return out


# ----------------------------------------------------------------------------- Y4 matching (Section 8.2)

HIST_EDGES = ((0, 0), (1, 4), (5, 12), (13, 24), (25, 40), (41, 48))


def competence_profile(ev3, ev5, tag):
    """Dev-panel profile of one initial adapter (48 attempts, cap 256)."""
    c3, n3 = ev3.arr(tag, "c"), ev3.n(tag)
    h = np.array([((c3 >= lo) & (c3 <= hi)).mean() for lo, hi in HIST_EDGES])
    b = bins_for(ev3, tag)
    cb = ev3.arr(tag, "c_b")
    base = [float(cb[b == j].mean() / 24) if (b == j).sum() >= MIN_BIN_N else None for j in range(len(PRIMARY_BINS))]
    return {"pass1_3": float(c3.mean() / n3), "pass1_5": float(ev5.arr(tag, "c").mean() / ev5.n(tag)),
            "hist": h.tolist(), "bin_base": base, "bin_n": [int((b == j).sum()) for j in range(len(PRIMARY_BINS))]}


def match(px, py, tol_p1=0.02, tol_tv=0.10, tol_base=0.03):
    """Section 8.2 rule. Returns (matched, residuals)."""
    tv = 0.5 * float(np.abs(np.array(px["hist"]) - np.array(py["hist"])).sum())
    d3 = abs(px["pass1_3"] - py["pass1_3"])
    d5 = abs(px["pass1_5"] - py["pass1_5"])
    db = [abs(a - b) for a, b in zip(px["bin_base"], py["bin_base"]) if a is not None and b is not None]
    res = {"d_pass1_3": d3, "d_pass1_5": d5, "tv": tv, "d_bin_base_max": max(db) if db else None,
           "n_bins_compared": len(db)}
    ok = d3 <= tol_p1 and d5 <= tol_p1 and tv <= tol_tv and bool(db) and max(db) <= tol_base
    return ok, res


# ----------------------------------------------------------------------------- problem structure (3 inputs)

def solution_forms(inputs, target):
    """Exact enumeration of the valid 2-operation chains of a 3-input problem. Returns (n_chains, forms), where a
    chain counts commutative operand orders once and a form is (op1, op2, side): side 'L' if the intermediate is the
    left operand of op2 ('L' for commutative op2)."""
    from .task import apply_op
    chains, forms = set(), set()
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            k = 3 - i - j
            a, b, c = inputs[i], inputs[j], inputs[k]
            for o1 in "+-*":
                r = apply_op(o1, a, b)
                for o2 in "+-*":
                    for x, y, side in ((r, c, "L"), (c, r, "R")):
                        if apply_op(o2, x, y) == target:
                            s1 = (min(a, b), o1, max(a, b)) if o1 in "+*" else (a, o1, b)
                            s2 = (min(x, y), o2, max(x, y)) if o2 in "+*" else (x, o2, y)
                            chains.add((s1, s2))
                            forms.add((o1, o2, "L" if o2 in "+*" else side))
    return len(chains), sorted(forms)


def form_of_text(text, n_inputs=3):
    """Form (op1, op2, side) of a successful 3-input completion, from the strict parser."""
    from .task import parse_trace
    ops, _ = parse_trace(text)
    if n_inputs != 3 or len(ops) < 2:
        return None
    (_, o1, _, c1), (a2, o2, _, _) = ops[0], ops[1]
    side = "L" if o2 in "+*" or a2 == c1 else "R"
    return (o1, o2, side)


def structure_analysis(ev, problems, forms, branches):
    """Exploratory: per-problem features against the excess loss_16, by init (pooled over methods and seeds);
    and the final policies' form preference. Hypothesis generation only.
    forms: {tag: {pid: Counter(form)}} over the successful attempts; branches: [{init, method, seed, ti, tf}].
    Form preference of a policy = share of its successful attempts (whole panel) that use a form; for a problem,
    the best preference over the problem's valid forms."""
    pmap = {p.pid: p for p in problems}
    feat = {}
    for pid in ev.pids:
        p = pmap[pid]
        nch, fl = solution_forms(p.inputs, p.target)
        feat[pid] = {"n_chains": nch, "n_forms": len(fl), "forms": fl,
                     "needs_mul": all("*" in f[:2] for f in fl) if fl else None,
                     "needs_sub": all("-" in f[:2] for f in fl) if fl else None,
                     "target_ge_100": p.target >= 100, "structure": p.structure}
    out = {"feature_counts": {"n_forms": dict(Counter(min(f["n_forms"], 4) for f in feat.values())),
                              "needs_mul": dict(Counter(str(f["needs_mul"]) for f in feat.values())),
                              "needs_sub": dict(Counter(str(f["needs_sub"]) for f in feat.values()))}}
    rows = defaultdict(list)
    pref_rows = []
    for br in branches:
        i, ti, tf = br["init"], br["ti"], br["tf"]
        e = per_problem(ev, ti, tf, K=16)
        b = bins_for(ev, ti)
        fin, ini = Counter(), Counter()
        for c in forms[tf].values():
            fin.update(c)
        for c in forms[ti].values():
            ini.update(c)
        nf, ni = sum(fin.values()) or 1, sum(ini.values()) or 1
        pref_rows.append({"init": i, "method": br["method"], "seed": br["seed"],
                          "final_top_form_share": fin.most_common(1)[0][1] / nf if fin else 0.0,
                          "initial_top_form_share": ini.most_common(1)[0][1] / ni if ini else 0.0,
                          "final_n_forms_used": len(fin), "initial_n_forms_used": len(ini)})
        for k, pid in enumerate(ev.pids):
            if b[k] < 0:
                continue
            f = feat[pid]
            pref_f = max((fin[x] / nf for x in map(tuple, f["forms"])), default=0.0)
            pref_i = max((ini[x] / ni for x in map(tuple, f["forms"])), default=0.0)
            for key, val in (("n_forms", min(f["n_forms"], 3)), ("needs_mul", f["needs_mul"]),
                             ("needs_sub", f["needs_sub"]), ("target_ge_100", f["target_ge_100"]),
                             ("structure", f["structure"]), ("form_preference", "all")):
                rows[(i, key, str(val))].append((float(e[k]), pref_f, pref_i))
    tab = []
    for (i, key, val), v in sorted(rows.items()):
        arr = np.array(v)
        if key == "form_preference":
            tab.append({"init": i, "feature": key, "n": len(arr),
                        "corr_excess_vs_pref_drop": float(np.corrcoef(arr[:, 0], arr[:, 2] - arr[:, 1])[0, 1]),
                        "corr_excess_vs_final_pref": float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])})
        else:
            tab.append({"init": i, "feature": key, "value": val, "n": len(arr),
                        "mean_excess": float(arr[:, 0].mean())})
    out["by_feature"] = tab
    out["form_preference"] = pref_rows
    return out


# ----------------------------------------------------------------------------- loading

def load_summaries(evals_dir, panel):
    out = {}
    for f in sorted(glob.glob(os.path.join(evals_dir, f"*.{panel}.summary.json"))):
        s = json.load(open(f))
        out[s["tag"]] = {r["pid"]: r for r in s["counts"]}
    return out


def counts_from_records(recs, split):
    """Rows with c, c_a, c_b and the cap-256 code string from full attempt records (A5 evaluations have no codes)."""
    from .extension import CODE
    by = {}
    for r in recs:
        by.setdefault(r["pid"], {})[int(r["attempt"])] = r
    rows = {}
    for pid, a in by.items():
        n = len(a)
        rew = [int(a[i]["reward"]) for i in range(n)]
        rows[pid] = {"pid": pid, "n": n, "c": sum(rew), "c_a": sum(rew[:split]) if split else "",
                     "c_b": sum(rew[split:]) if split else "",
                     "codes": "".join(CODE[a[i]["error"]] for i in range(n))}
    return rows
