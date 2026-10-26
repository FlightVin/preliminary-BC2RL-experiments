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
    assert st["xa"]["reject:duplicate"] >= 1      # a rejection shifts the rng stream, so later draws differ
    rep, _ = draw_fresh(5, g, {"xa": (3, 40)}, set(seen))
    assert [p.key for p in rep["xa"]] == keys
    for p in again["xa"]:
        assert task.verify(p.inputs, p.target, task.completion_text(p))["reward"] == 1


def test_x7_replication_rule(monkeypatch):
    """X7 uses the X5 estimator with the new seeds; REPLICATED needs the lower limit > 0 and every seed Delta_s > 0."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("x7", "scripts/x7_replicate.py")
    x7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(x7)
    monkeypatch.setattr(X, "SEEDS", x7.NEW_SEEDS)

    def change(i, m, p):
        return np.where((i == "extended") & (p < 0.5), p * 0.4, p)

    prim, ds = x7.analyse(synth_panel(np.random.default_rng(4), change=change), 400, x7.NEW_SEEDS)
    assert set(ds) == set(x7.NEW_SEEDS)
    assert x7.decide_x7(prim["ci95"][0], prim["ci95"][1], ds) == "REPLICATED"
    prim0, ds0 = x7.analyse(synth_panel(np.random.default_rng(5)), 400, x7.NEW_SEEDS)
    assert x7.decide_x7(prim0["ci95"][0], prim0["ci95"][1], ds0) in ("NOT REPLICATED", "CONTRADICTED")
    assert x7.decide_x7(0.1, 0.3, {44: 0.2, 55: -0.01, 66: 0.3}) == "NOT REPLICATED"
    assert x7.decide_x7(0.1, 0.3, {44: 0.2, 55: float("nan"), 66: 0.3}) == "NOT REPLICATED"


def test_x7_configs_match_frozen_protocol():
    """Real X7 configs: same settings as the seed-33 frozen configs except seed, run_id, stream seed and init path."""
    import json
    wl = json.load(open("configs/worklists/x7.json"))
    assert wl["name"] == "x7" and len(wl["tasks"]) == 15
    for s in (44, 55, 66):
        b, b0 = json.load(open(f"configs/ext/x7/bc-broad-s{s}.json")), json.load(open("configs/main/bc-broad-s33.yaml"))
        assert {k: v for k, v in b.items() if k not in ("run_id", "seed")} == \
            {k: v for k, v in b0.items() if k not in ("run_id", "seed")} and b["seed"] == s
        for i, st in (("light", 16), ("extended", 128)):
            for m in ("grpo", "filtered"):
                o, o0 = json.load(open(f"configs/ext/x7/on-{i}-{m}-s{s}.json")), json.load(open(f"configs/main/on-{i}-{m}-s33.yaml"))
                strip = lambda c: {**{k: v for k, v in c.items() if k not in ("run_id", "seed", "online")},
                                   "online": {k: v for k, v in c["online"].items() if k not in ("stream_seed", "init")}}
                assert strip(o) == strip(o0) and o["seed"] == s and o["online"]["stream_seed"] == s
                assert o["online"]["init"]["adapter"] == f"runs/bc2rl-s1-bc-broad-s{s}/adapters/step{st:04d}.safetensors"
                ev = json.load(open(f"configs/ext/x7/xeval-{i}-{m}-s{s}.json"))
                assert ev["xeval"]["checkpoints"][0]["tag"] == X.final_tag(i, m, s)
                assert ev["xeval"]["checkpoints"][0]["adapter"] == f"runs/{o['run_id']}/adapters/round0256.safetensors"
                assert {k: v for k, v in ev["xeval"].items() if k != "checkpoints"} == \
                    {k: v for k, v in json.load(open("configs/ext/xeval-w2.json"))["xeval"].items() if k != "checkpoints"}
        ev = json.load(open(f"configs/ext/x7/xeval-init-s{s}.json"))
        assert [c["tag"] for c in ev["xeval"]["checkpoints"]] == [X.init_tag("light", s), X.init_tag("extended", s)]
