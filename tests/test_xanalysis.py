"""Tests for the A5 confirmation analysis and fresh-data draw (synthetic data only; no model)."""
from math import comb

import numpy as np
import pytest

from bc2rl import metrics, task
from bc2rl import xanalysis as X
from bc2rl.extension import draw_fresh


def synth_panel(rng, n=768, change=None):
    """Initial p: light ~ Beta(0.5, 3), extended ~ Beta(0.9, 1.4) (different profiles). change(i, m, p) -> final p."""
    counts = {}
    pids = [f"x-{j:05d}" for j in range(n)]
    for s in X.SEEDS:
        for i, (a, b) in (("light", (0.5, 3.0)), ("extended", (0.9, 1.4))):
            p = rng.beta(a, b, n)
            ca, cb = rng.binomial(24, p), rng.binomial(24, p)
            counts[X.init_tag(i, s)] = {q: {"n": 48, "c": int(x + y), "c_a": int(x), "c_b": int(y)}
                                        for q, x, y in zip(pids, ca, cb)}
            for m in X.METHODS:
                pf = p if change is None else change(i, m, p)
                cf = rng.binomial(32, pf)
                counts[X.final_tag(i, m, s)] = {q: {"n": 32, "c": int(x), "c_a": "", "c_b": ""}
                                                for q, x in zip(pids, cf)}
    return X.Panel(counts)


def test_estimators_match_metrics():
    for n in (24, 32, 48):
        for c in range(n + 1):
            for k in (1, 4, 16, 24):
                if k <= n:
                    assert X.loss_k(n, [c], k)[0] == pytest.approx(1 - metrics.pass_at_k(n, c, k), abs=1e-12)
                    assert X.rel_k(n, [c], k)[0] == pytest.approx(metrics.pass_hat_k(n, c, k), abs=1e-12)


def test_loss_estimator_unbiased():
    p, n, k = 0.3, 24, 16
    exp = sum(comb(n, c) * p ** c * (1 - p) ** (n - c) * X.loss_k(n, [c], k)[0] for c in range(n + 1))
    assert exp == pytest.approx((1 - p) ** k, rel=1e-9)


def test_null_no_bias_and_coverage():
    rng = np.random.default_rng(1)
    covered, means = 0, []
    for _ in range(12):
        pan = synth_panel(rng)
        r = X.primary(pan, n_boot=400)
        means.append(r["mean_delta"])
        covered += r["ci95"][0] <= 0 <= r["ci95"][1]
        assert r["decision"] in ("NOT CONFIRMED", "CONFIRMED", "CONTRADICTED")
    assert abs(np.mean(means)) < 0.01
    assert covered >= 9


def test_naive_baseline_is_biased_but_design_is_not():
    """Selecting and estimating the baseline from the same attempts shows regression to the mean; the split does not."""
    rng = np.random.default_rng(2)
    pan = synth_panel(rng)
    i, m, s = "light", "grpo", 11
    b = X.bin_of(pan.ca[X.init_tag(i, s)], X.PRIMARY_BINS)
    msk = b >= 0
    final = X.loss_k(32, pan.c[X.final_tag(i, m, s)][msk], 16).mean()
    naive = final - X.loss_k(24, pan.ca[X.init_tag(i, s)][msk], 16).mean()
    split = final - X.loss_k(24, pan.cb[X.init_tag(i, s)][msk], 16).mean()
    assert abs(naive) > 0.03
    assert abs(split) < 0.03


def test_planted_effect_detected():
    rng = np.random.default_rng(3)

    def change(i, m, p):
        return np.where((i == "extended") & (p < 0.5), p * 0.4, p)

    r = X.primary(synth_panel(rng, change=change), n_boot=400)
    assert r["mean_delta"] > 0.05
    assert r["decision"] == "CONFIRMED"
    assert r["pairs_positive"] == 6


def test_bins_and_weights():
    c = np.array([0, 1, 2, 3, 5, 6, 11, 12, 24])
    assert X.bin_of(c, X.PRIMARY_BINS).tolist() == [-1, 0, 0, 1, 1, 2, 2, -1, -1]
    w = X.bin_weights({("a", 1): c}, X.PRIMARY_BINS)
    assert w.tolist() == pytest.approx([2 / 6, 2 / 6, 2 / 6])


def test_panel_requires_same_problems():
    with pytest.raises(ValueError):
        X.Panel({"a": {"p1": {"n": 1, "c": 0}}, "b": {"p2": {"n": 1, "c": 0}}})


def test_draw_fresh_excludes_seen_and_is_deterministic():
    g = task.GenConfig()
    first, _ = draw_fresh(5, g, {"xa": (3, 40)}, set())
    seen = {p.key for p in first["xa"][:20]}
    again, st = draw_fresh(5, g, {"xa": (3, 40)}, set(seen))
    keys = [p.key for p in again["xa"]]
    assert not seen & set(keys)
    assert len(set(keys)) == 40
    assert st["xa"]["reject:duplicate"] >= 20
    rep, _ = draw_fresh(5, g, {"xa": (3, 40)}, set(seen))
    assert [p.key for p in rep["xa"]] == keys
    for p in again["xa"]:
        assert task.verify(p.inputs, p.target, task.completion_text(p))["reward"] == 1
