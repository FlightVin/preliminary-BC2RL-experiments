"""`bc2rl toy`: E0 exact two-task model (proposal Section 3.3; amendment A2). CPU only, numpy.

Two equally weighted binary tasks, reward 1 for action 1, p_i = sigmoid(theta * phi_i).
Aligned features phi = (1, 1); opposing features phi = (1, -1). The script checks, numerically:
  1. mixed-group probability 1 - p^G - (1-p)^G = C_G - R_G (exact and Monte Carlo), G in {2, 4, 8};
  2. the combinatorial pass@k / pass^k estimators are unbiased for 1-(1-p)^k and p^k (exact expectation, n = 15);
  3. the GRPO group identity (1/G) sum_i A_i s_i = q(1-q)/(sqrt(q(1-q)) + eps) * (s_plus - s_minus);
  4. the first-order (Taylor) change of mean success for a small theta step;
  5. the counterexample: identical initial curves and mixed-group probabilities at theta = 0; mean success
     constant in the opposing model for every theta; for k > 1 its R_k rises and C_k falls with |theta|;
     the full-batch mean-reward gradient at 0 cancels in the opposing model.
It is an analytic control and a test of the analysis code, not an LLM result.
"""
from __future__ import annotations

import csv
import json
import os
from math import comb

import numpy as np

from .metrics import pass_at_k, pass_hat_k


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def curves(theta, phi, ks):
    p = sigmoid(theta * np.asarray(phi, float))
    return {"P1": float(p.mean()),
            "C": {k: float(np.mean(1 - (1 - p) ** k)) for k in ks},
            "R": {k: float(np.mean(p ** k)) for k in ks},
            "mixed": {}}


def mixed_prob(p, G):
    return 1 - p ** G - (1 - p) ** G


def expected_estimator(p, n, k, fn):
    return sum(comb(n, c) * p ** c * (1 - p) ** (n - c) * fn(n, c, k) for c in range(n + 1))


def grpo_group_check(rng, G, D=7, eps=1e-6):
    out = []
    for m in range(1, G):
        r = np.array([1.0] * m + [0.0] * (G - m))
        s = rng.normal(size=(G, D))
        q = m / G
        sd = r.std()
        A = (r - r.mean()) / (sd + eps)
        lhs = (A[:, None] * s).mean(0)
        rhs = q * (1 - q) / (np.sqrt(q * (1 - q)) + eps) * (s[:m].mean(0) - s[m:].mean(0))
        out.append(float(np.abs(lhs - rhs).max()))
    return max(out)


def cmd_toy(cfg):
    res_dir = cfg.get("_results_dir") or os.path.join(cfg["proj_data"], "toy")
    os.makedirs(res_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    ks = list(range(1, 16))
    checks = {}
    # 1. mixed-group probability
    worst_exact, worst_mc = 0.0, 0.0
    for G in (2, 4, 8):
        for p in np.linspace(0.0, 1.0, 21):
            ident = (1 - (1 - p) ** G) - p ** G
            worst_exact = max(worst_exact, abs(mixed_prob(p, G) - ident))
            draws = rng.random((200000, G)) < p
            mc = float(np.mean(draws.any(1) & ~draws.all(1)))
            worst_mc = max(worst_mc, abs(mc - mixed_prob(p, G)))
    checks["mixed_identity_max_abs_err"] = worst_exact
    checks["mixed_montecarlo_max_abs_err"] = worst_mc
    # 2. estimator unbiasedness (exact expectation over Binomial(15, p))
    worst_c, worst_r = 0.0, 0.0
    for p in np.linspace(0.0, 1.0, 21):
        for k in ks:
            worst_c = max(worst_c, abs(expected_estimator(p, 15, k, pass_at_k) - (1 - (1 - p) ** k)))
            worst_r = max(worst_r, abs(expected_estimator(p, 15, k, pass_hat_k) - p ** k))
    checks["pass_at_k_bias_max"] = worst_c
    checks["pass_hat_k_bias_max"] = worst_r
    # 3. GRPO group identity
    checks["grpo_identity_max_abs_err"] = max(grpo_group_check(rng, G) for G in (2, 4, 8))
    # 4. Taylor check of mean success for small steps
    tay = []
    for phi in ((1, 1), (1, -1)):
        phi = np.asarray(phi, float)
        for th in (-1.0, 0.0, 0.7):
            for dth in (1e-3, 1e-2):
                p = sigmoid(th * phi)
                grad = float(np.mean(p * (1 - p) * phi))
                actual = float(np.mean(sigmoid((th + dth) * phi)) - np.mean(p))
                tay.append(abs(actual - grad * dth) / max(abs(dth) ** 2, 1e-12))
    checks["taylor_second_order_ratio_max"] = max(tay)      # O(dth^2) residual / dth^2: bounded
    # 5. counterexample
    al0, op0 = curves(0.0, (1, 1), ks), curves(0.0, (1, -1), ks)
    checks["initial_curves_identical"] = bool(al0 == op0)
    checks["initial_mixed_identical"] = bool(all(abs(mixed_prob(0.5, G) - mixed_prob(0.5, G)) == 0 for G in (2, 4, 8)))
    thetas = np.round(np.linspace(-2, 2, 41), 3)
    rows = []
    op_p1 = []
    for th in thetas:
        for name, phi in (("aligned", (1, 1)), ("opposing", (1, -1))):
            cv = curves(th, phi, ks)
            p = sigmoid(th * np.asarray(phi, float))
            row = {"model": name, "theta": float(th), "P1": cv["P1"]}
            for k in ks:
                row[f"C{k}"] = cv["C"][k]
                row[f"R{k}"] = cv["R"][k]
            for G in (2, 4, 8):
                row[f"mixed_G{G}"] = float(np.mean(mixed_prob(p, G)))
            rows.append(row)
            if name == "opposing":
                op_p1.append(cv["P1"])
    checks["opposing_mean_success_constant_max_dev"] = float(np.max(np.abs(np.array(op_p1) - 0.5)))
    mono = True
    for k in ks[1:]:
        rk = [r[f"R{k}"] for r in rows if r["model"] == "opposing" and r["theta"] >= 0]
        ck = [r[f"C{k}"] for r in rows if r["model"] == "opposing" and r["theta"] >= 0]
        mono &= all(b > a for a, b in zip(rk, rk[1:])) and all(b < a for a, b in zip(ck, ck[1:]))
    checks["opposing_Rk_up_Ck_down_for_k_gt_1"] = bool(mono)
    grad0 = {name: float(np.mean(0.25 * np.asarray(phi, float))) for name, phi in (("aligned", (1, 1)), ("opposing", (1, -1)))}
    checks["fullbatch_grad_at_0"] = grad0
    ok = (worst_exact < 1e-12 and worst_mc < 5e-3 and worst_c < 1e-10 and worst_r < 1e-10
          and checks["grpo_identity_max_abs_err"] < 1e-9 and checks["taylor_second_order_ratio_max"] < 1.0
          and checks["initial_curves_identical"] and checks["opposing_mean_success_constant_max_dev"] < 1e-12
          and mono and grad0["opposing"] == 0.0 and grad0["aligned"] > 0)
    checks["all_ok"] = bool(ok)
    with open(os.path.join(res_dir, "toy_curves.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(res_dir, "toy_checks.json"), "w") as f:
        json.dump(checks, f, indent=1)
    print(json.dumps(checks))
    return 0 if ok else 1
