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


def bootstrap_indices(horizons, n_boot=10000, seed=0):
    """Horizon-stratified resampling indices, shared by every condition (paired bootstrap)."""
    horizons = np.asarray(horizons)
    rng = np.random.default_rng(seed)
    out = {}
    for h in sorted(set(horizons.tolist())):
        ix = np.flatnonzero(horizons == h)
        out[h] = ix[rng.integers(0, len(ix), size=(n_boot, len(ix)))]
    return out


def boot_weighted(values, idx):
    """(n_boot,) horizon-weighted means of one per-problem array under shared indices."""
    v = np.asarray(values, float)
    return np.mean([v[ix].mean(axis=1) for ix in idx.values()], axis=0)


def paired_bootstrap(per_cond: dict, horizons, n_boot=10000, seed=0, diffs=()):
    """Paired, horizon-stratified bootstrap over whole problems.

    per_cond: {name: per-problem metric array}, all aligned to the same problem order.
    diffs: iterable of (a, b) names; reports CI of weighted(a) - weighted(b).
    Returns {name or 'a - b': (point, lo95, hi95)}. Same resampled indices for every condition.
    """
    idx = bootstrap_indices(horizons, n_boot, seed)
    samples = {k: boot_weighted(v, idx) for k, v in per_cond.items()}
    out = {}
    for k, v in per_cond.items():
        pt = horizon_weighted(v, horizons)[0]
        out[k] = (pt, float(np.percentile(samples[k], 2.5)), float(np.percentile(samples[k], 97.5)))
    for a, b in diffs:
        pt = horizon_weighted(per_cond[a], horizons)[0] - horizon_weighted(per_cond[b], horizons)[0]
        d = samples[a] - samples[b]
        out[f"{a} - {b}"] = (pt, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
    return out


def zero_event_upper_bound(n_problems: int, alpha: float = 0.05) -> float:
    """One-sided (1-alpha) upper bound on a Bernoulli rate after 0 successes in n trials."""
    return 1.0 - alpha ** (1.0 / n_problems)


def quartet_indicator(successes_by_attempt, quartet=(0, 1, 2, 3)):
    """Preselected-quartet all-success indicator for one problem (attempt-indexed 0/1 list)."""
    return int(all(successes_by_attempt[i] for i in quartet))
