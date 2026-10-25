"""Amendment-A2 analysis: retention split, contrast bands, audit rows (synthetic panels)."""
import numpy as np

from bc2rl.analysis import _p_zero, audit_rows, contrast, coverage, curves_for, retention


def panel(att, h):
    pids = [f"p{i:03d}" for i in range(len(h))]
    recs = [{"pid": p, "attempt": a, "reward": int(att[i, a]), "n_inputs": int(h[i]), "error": "success" if att[i, a] else "wrong_final",
             "format_valid": True, "finish": "stop"} for i, p in enumerate(pids) for a in range(att.shape[1])]
    return {"spec": {}, "pids": pids, "h": np.array(h), "att": np.array(att), "recs": recs}


def test_retention_uses_independent_split():
    h = [3] * 20 + [5] * 20
    ia = np.zeros((40, 15), int)
    ia[:14, :7] = 1                     # 14 consistent at 3 inputs (first 7 all success)
    ia[:14, 7:] = 0                     # independent baseline: all failures -> baseline 0
    fa = np.ones((40, 15), int)
    rows, trans, d = retention(panel(ia, h), panel(fa, h), n_boot=200)
    r3 = next(r for r in rows if r["n_inputs"] == 3)
    assert r3["n_consistent"] == 14 and r3["status"] == "ok"
    assert r3["pass@1_base"] == 0.0 and r3["pass@1_final"] == 1.0
    r5 = next(r for r in rows if r["n_inputs"] == 5)
    assert r5["n_consistent"] == 0 and r5["status"] == "none"
    assert "pass@9_diff" not in r3 and "pass^8_diff" in r3


def test_contrast_bands_cover_point_and_simultaneous_wider():
    rng = np.random.default_rng(0)
    h = np.array([3] * 30 + [5] * 30 + [10] * 30)
    a = (rng.random((90, 15)) < 0.4).astype(int)
    b = (rng.random((90, 15)) < 0.3).astype(int)
    rows = contrast(a, b, h, 500, 15)
    assert len(rows) == 30
    for r in rows:
        assert r["sim_lo"] <= r["diff"] <= r["sim_hi"]
    assert np.mean([(r["sim_hi"] - r["sim_lo"]) >= (r["pw_hi"] - r["pw_lo"]) - 1e-12 for r in rows]) > 0.5


def test_curves_and_audit():
    att = np.array([[1] * 15, [0] * 15, [1, 0] * 7 + [1]])
    cv = curves_for(att, np.array([3, 3, 3]), 15)
    assert cv[("pass^", 15)].tolist() == [1.0, 0.0, 0.0]
    assert abs(cv[("pass@", 1)][2] - 8 / 15) < 1e-12
    rows = audit_rows("x", panel(att, [3, 3, 3]))
    assert rows[0]["problems_all_success"] == 1 and rows[0]["frac_success"] == (15 + 8) / 45


def test_p_zero_matches_closed_form_product():
    # B(a, b + m) / B(a, b) = prod_{j<m} (b + j) / (a + b + j) for integer m (independent formula).
    for c in (0, 1, 7, 15):
        a, b = c + 0.5, 15 - c + 0.5
        ref = float(np.prod([(b + j) / (a + b + j) for j in range(15)]))
        assert abs(_p_zero(c, 15, 15) - ref) < 1e-12
    assert _p_zero(15, 15, 15) < 1e-3 < _p_zero(1, 15, 15) < _p_zero(0, 15, 15) < 1


def test_coverage_counts_and_calibration_under_no_change():
    rng = np.random.default_rng(1)
    h = [3] * 60 + [5] * 60
    obs, exp, inside = [], [], 0
    for rep in range(200):
        # true rates drawn from the reference prior Beta(1/2, 1/2): the reference is then calibrated
        p = np.r_[rng.beta(0.5, 0.5, 60), np.zeros(60)]
        ia = (rng.random((120, 15)) < p[:, None]).astype(int)
        fa = (rng.random((120, 15)) < p[:, None]).astype(int)    # no change: same success rates
        rows = coverage(panel(ia, h), panel(fa, h), n_sim=500, seed=rep)
        r3 = next(r for r in rows if r["n_inputs"] == 3)
        if rep == 0:
            assert r3["lost_n"] == int((ia[:60].sum(1) > 0).sum())
            assert r3["lost"] == int(((ia[:60].sum(1) > 0) & (fa[:60].sum(1) == 0)).sum())
            r5 = next(r for r in rows if r["n_inputs"] == 5)
            assert r5["lost_n"] == 0 and r5["gained_n"] == 60 and r5["gained"] == 0
        obs.append(r3["lost"]); exp.append(r3["lost_expected"])
        inside += r3["lost_lo"] <= r3["lost"] <= r3["lost_hi"]
    assert abs(np.mean(obs) - np.mean(exp)) < 0.15 * np.mean(exp)
    assert inside >= 180                    # nominal 95% band covers >= 90% of 200 no-change replicates
    rows = coverage(panel(ia, h), panel(np.zeros_like(fa), h), n_sim=2000)    # everything lost
    r3 = next(r for r in rows if r["n_inputs"] == 3)
    assert r3["lost"] == r3["lost_n"] > r3["lost_hi"]
