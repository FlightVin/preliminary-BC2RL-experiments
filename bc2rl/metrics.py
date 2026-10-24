"""Repeated-sampling estimators and uncertainty (PROJECT.md "Evaluation and interpretation").

With c successes among n attempts on one problem:
  pass@k = 1 - C(n-c, k) / C(n, k)      (at least one success in k draws without replacement)
  pass^k = C(c, k) / C(n, k)            (all k draws succeed); C(a, k) = 0 for a < k
Aggregate: mean within each horizon, then equal weight across horizons.
"""
from __future__ import annotations

from math import comb

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    if k > n:
        raise ValueError("k > n")
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    if k > n:
        raise ValueError("k > n")
    return comb(c, k) / comb(n, k)       # math.comb(c, k) == 0 when c < k


METRICS = {
    "pass@1": lambda n, c: c / n,
    "pass@4": lambda n, c: pass_at_k(n, c, 4),
    "pass^4": lambda n, c: pass_hat_k(n, c, 4),
    "pass^2": lambda n, c: pass_hat_k(n, c, 2),
    "mixed4": lambda n, c: pass_at_k(n, c, 4) - pass_hat_k(n, c, 4),
}


def per_problem(n, c, metric):
    f = METRICS[metric]
    return np.array([f(int(nn), int(cc)) for nn, cc in zip(n, c)], dtype=float)


def horizon_weighted(values, horizons):
    """Mean within each horizon, then equal weight across horizons. Returns (overall, {h: mean})."""
    values = np.asarray(values, float)
    horizons = np.asarray(horizons)
    by = {int(h): float(values[horizons == h].mean()) for h in sorted(set(horizons.tolist()))}
    return float(np.mean(list(by.values()))), by


def paired_bootstrap(per_cond: dict, horizons, n_boot=10000, seed=0, diffs=()):
    """Paired, horizon-stratified bootstrap over whole problems.

    per_cond: {name: per-problem metric array}, all aligned to the same problem order.
    diffs: iterable of (a, b) names; reports CI of weighted(a) - weighted(b).
    Returns {name or 'a - b': (point, lo95, hi95)}. Same resampled indices for every condition.
    """
    horizons = np.asarray(horizons)
    hs = sorted(set(horizons.tolist()))
    idx_by_h = {h: np.flatnonzero(horizons == h) for h in hs}
    rng = np.random.default_rng(seed)
    names = list(per_cond)
    arr = {k: np.asarray(v, float) for k, v in per_cond.items()}
    samples = {k: np.empty(n_boot) for k in names}
    for b in range(n_boot):
        picks = {h: rng.choice(ix, size=len(ix), replace=True) for h, ix in idx_by_h.items()}
        for k in names:
            samples[k][b] = np.mean([arr[k][picks[h]].mean() for h in hs])
    out = {}
    for k in names:
        pt = horizon_weighted(arr[k], horizons)[0]
        out[k] = (pt, float(np.percentile(samples[k], 2.5)), float(np.percentile(samples[k], 97.5)))
    for a, b in diffs:
        pt = horizon_weighted(arr[a], horizons)[0] - horizon_weighted(arr[b], horizons)[0]
        d = samples[a] - samples[b]
        out[f"{a} - {b}"] = (pt, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
    return out


def zero_event_upper_bound(n_problems: int, alpha: float = 0.05) -> float:
    """One-sided (1-alpha) upper bound on a Bernoulli rate after 0 successes in n trials."""
    return 1.0 - alpha ** (1.0 / n_problems)


def quartet_indicator(successes_by_attempt, quartet=(0, 1, 2, 3)):
    """Preselected-quartet all-success indicator for one problem (attempt-indexed 0/1 list)."""
    return int(all(successes_by_attempt[i] for i in quartet))
