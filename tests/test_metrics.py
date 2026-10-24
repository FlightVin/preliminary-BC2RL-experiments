from itertools import combinations

import numpy as np

from bc2rl import metrics


def brute(n, c, k, allsucc):
    att = [1] * c + [0] * (n - c)
    subs = list(combinations(range(n), k))
    if allsucc:
        return sum(all(att[i] for i in s) for s in subs) / len(subs)
    return sum(any(att[i] for i in s) for s in subs) / len(subs)


def test_estimators_match_enumeration():
    for n in (4, 6, 8):
        for c in range(n + 1):
            assert abs(metrics.pass_at_k(n, c, 4) - brute(n, c, 4, False)) < 1e-12
            assert abs(metrics.pass_hat_k(n, c, 4) - brute(n, c, 4, True)) < 1e-12
            assert abs(metrics.pass_hat_k(n, c, 2) - brute(n, c, 2, True)) < 1e-12


def test_n15_edge_values():
    assert metrics.pass_at_k(15, 0, 4) == 0 and metrics.pass_at_k(15, 15, 4) == 1
    assert metrics.pass_hat_k(15, 3, 4) == 0
    assert abs(metrics.pass_hat_k(15, 4, 4) - 1 / 1365) < 1e-15
    assert abs(metrics.pass_at_k(15, 12, 4) - 1.0) < 1e-12        # C(3,4)=0


def test_not_plugin_power():
    # c/n = 0.5 at n=4: plug-in p^4 = 0.0625 but unbiased pass^4 = C(2,4)/C(4,4) = 0
    assert metrics.pass_hat_k(4, 2, 4) == 0.0


def test_horizon_weighting():
    vals = [1, 1, 1, 0, 0, 0, 0, 0]
    hs = [3, 3, 3, 5, 5, 10, 10, 10]
    overall, by = metrics.horizon_weighted(vals, hs)
    assert by == {3: 1.0, 5: 0.0, 10: 0.0} and abs(overall - 1 / 3) < 1e-12


def test_paired_bootstrap_identical_conditions_zero_diff():
    rng = np.random.default_rng(0)
    v = rng.random(30)
    hs = [3] * 10 + [5] * 10 + [10] * 10
    out = metrics.paired_bootstrap({"a": v, "b": v.copy()}, hs, n_boot=200, diffs=[("a", "b")])
    assert out["a - b"] == (0.0, 0.0, 0.0)
    lo, hi = out["a"][1], out["a"][2]
    assert lo <= out["a"][0] <= hi


def test_rare_event_bound_matches_proposal():
    assert abs(metrics.zero_event_upper_bound(288) - 0.0103) < 2e-4
    assert abs(metrics.zero_event_upper_bound(96) - 0.0307) < 2e-4


def test_config_floats_parse_as_numbers(tmp_path):
    import json as _json
    from bc2rl.runtime import load_config
    p = tmp_path / "c.yaml"
    _json.dump({"run_id": "x", "proj_data": "/tmp", "online": {"lr": 5e-06}, "bc": {"lr": 1e-05}}, open(p, "w"))
    c = load_config(str(p))
    assert isinstance(c["online"]["lr"], float) and c["online"]["lr"] == 5e-06 and c["bc"]["lr"] == 1e-05
