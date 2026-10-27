"""A6 Section 12 (GSPO stage) loss tests: masks and stop-token handling, clipping branches, first-update agreement of
GSPO and length-normalized GRPO, and fp64 gradient checks. CPU only, no model."""
import math

import torch

from bc2rl.losses import grpo_len_loss_sum, grpo_loss_sum, gspo_loss_sum

torch.manual_seed(0)


def batch(dtype=torch.float64, B=6, T=7):
    """Rows of different length; row 5 is a full-length (truncated) row; row 3 has A = 0."""
    lens = torch.tensor([1, 3, 7, 4, 2, 7])
    mask = (torch.arange(T).unsqueeze(0) < lens.unsqueeze(1)).to(dtype)
    logp = (-torch.rand(B, T, dtype=dtype) * 3).requires_grad_(True)
    adv = torch.tensor([1.2, -0.4, 0.7, 0.0, -1.5, 0.9], dtype=dtype)
    return logp, mask, adv, lens


def test_first_update_gradients_agree_and_equal_closed_form():
    logp, mask, adv, lens = batch()
    Q = 8
    g = {}
    for name, fn in (("len", grpo_len_loss_sum), ("gspo", gspo_loss_sum)):
        lp = logp.detach().clone().requires_grad_(True)
        ls, ratio, st = fn(lp, lp.detach(), adv, mask, 0.2, 0.27)
        assert torch.all(ratio == 1)
        assert st["outside"] == 0 and st["clipped"] == 0
        (ls / Q).backward()
        g[name] = lp.grad
    closed = -(adv.unsqueeze(1) * mask / lens.unsqueeze(1).to(mask.dtype)) / Q
    assert torch.allclose(g["len"], closed, atol=1e-15)
    assert torch.allclose(g["gspo"], closed, atol=1e-15)


def test_masked_positions_have_no_effect():
    # Positions after the stop token (padding) never change the loss or receive a gradient.
    logp, mask, adv, _ = batch()
    beh = logp.detach() + 0.01 * torch.randn_like(logp)
    for fn in (grpo_len_loss_sum, gspo_loss_sum):
        lp1 = logp.detach().clone().requires_grad_(True)
        l1, _, _ = fn(lp1, beh, adv, mask, 0.2, 0.2)
        l1.backward()
        lp2 = (logp.detach() + (1 - mask) * 5.0).requires_grad_(True)
        beh2 = beh + (1 - mask) * -7.0
        l2, _, _ = fn(lp2, beh2, adv, mask, 0.2, 0.2)
        l2.backward()
        assert torch.allclose(l1, l2)
        assert torch.all(lp1.grad * (1 - mask) == 0) and torch.allclose(lp1.grad, lp2.grad)


def test_length_counts_stop_token_and_truncated_rows_use_cap():
    # |y_i| = mask row sum: the stop token is a completion token (modeling.sample keeps it); a truncated row has
    # every position up to the cap. The GSPO log-ratio is the masked mean.
    logp, mask, adv, lens = batch()
    beh = logp.detach() - 0.3 * mask
    _, s, _ = gspo_loss_sum(logp, beh, adv, mask, 1.0, 1.0)
    assert torch.allclose(s, torch.full_like(s, math.exp(0.3)))
    assert int(mask[5].sum()) == 7


def test_gspo_sequence_ratio_definition():
    logp, mask, adv, lens = batch()
    beh = logp.detach() + torch.randn_like(logp) * 0.05
    _, s, _ = gspo_loss_sum(logp, beh, adv, mask, 1.0, 1.0)
    for i in range(6):
        n = int(lens[i])
        ref = torch.exp((logp[i, :n] - beh[i, :n]).sum() / n)
        assert torch.allclose(s[i], ref)


def test_gspo_clipping_branches_asymmetric():
    # one token per row; s = exp(logp - beh)
    mask = torch.ones(4, 1, dtype=torch.float64)
    beh = torch.zeros(4, 1, dtype=torch.float64)
    e_lo, e_hi = 0.1, 0.2
    vals = torch.tensor([[math.log(1.25)], [math.log(0.85)], [math.log(0.85)], [math.log(1.15)]], dtype=torch.float64)
    adv = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64)
    lp = vals.clone().requires_grad_(True)
    ls, s, st = gspo_loss_sum(lp, beh, adv, mask, e_lo, e_hi)
    ls.backward()
    # row 0: A > 0, s = 1.25 > 1.2 -> clipped, zero gradient; row 1: A < 0, s = 0.85 < 0.9 -> clipped
    # row 2: A > 0, s = 0.85 outside the band but not clipped (min picks the unclipped term) -> gradient
    # row 3: A < 0, s = 1.15 inside the band -> gradient
    assert lp.grad[0, 0] == 0 and lp.grad[1, 0] == 0
    assert lp.grad[2, 0] != 0 and lp.grad[3, 0] != 0
    assert st["outside"] == 3 and st["clipped"] == 2 and st["units"] == 4
    assert st["clipped_tokens"] == 2 and st["unit_tokens"] == 4
    # objective value: -(sum_i min(s A, clip(s) A))
    exp = -(1.2 * 1 + (-1) * 0.9 + 0.85 * 1 + (-1) * 1.15)
    assert abs(float(ls) - exp) < 1e-12


def test_grpo_len_token_clipping_and_per_row_weights():
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    beh = torch.zeros(2, 3, dtype=torch.float64)
    lp = torch.tensor([[math.log(1.5), 0.0, 9.0], [0.0, math.log(0.5), 0.0]], dtype=torch.float64,
                      requires_grad=True)
    adv = torch.tensor([1.0, -2.0], dtype=torch.float64)
    ls, _, st = grpo_len_loss_sum(lp, beh, adv, mask, 0.2, 0.2)
    ls.backward()
    # row 0 token 0: A > 0, r = 1.5 > 1.2 -> clipped; token 1 r = 1 -> grad -A / 2; token 2 is padding
    # row 1 token 1: A < 0, r = 0.5 < 0.8 -> clipped; tokens 0, 2 -> grad -A / 3 * r
    assert lp.grad[0, 0] == 0 and abs(float(lp.grad[0, 1]) + 0.5) < 1e-12 and lp.grad[0, 2] == 0
    assert lp.grad[1, 1] == 0 and abs(float(lp.grad[1, 0]) - 2 / 3) < 1e-12
    assert st["units"] == 5 and st["outside"] == 2 and st["clipped"] == 2
    exp = -((1.2 + 1.0) / 2 + (-2 * 1 + -2 * 0.8 + -2 * 1) / 3)
    assert abs(float(ls) - exp) < 1e-12


def test_zero_advantage_rows_are_not_units_and_have_no_gradient():
    logp, mask, adv, _ = batch()
    beh = logp.detach() + 0.5
    for fn in (grpo_len_loss_sum, gspo_loss_sum):
        lp = logp.detach().clone().requires_grad_(True)
        ls, _, st = fn(lp, beh, adv, mask, 0.01, 0.01)
        ls.backward()
        assert torch.all(lp.grad[3] == 0)
    _, _, st = gspo_loss_sum(logp, beh, adv, mask, 0.01, 0.01)
    assert st["units"] == 5


def test_gradcheck_fp64_off_the_clip_boundary():
    logp, mask, adv, _ = batch()
    beh = logp.detach() + 0.02 * torch.randn_like(logp)
    for fn in (grpo_len_loss_sum, gspo_loss_sum):
        f = lambda x: fn(x, beh, adv, mask, 0.3, 0.3)[0]
        assert torch.autograd.gradcheck(f, (logp.detach().clone().requires_grad_(True),), eps=1e-6, atol=1e-7)
        f2 = lambda x: fn(x, beh, adv, mask, 1e-3, 1e-3)[0]          # many units clipped
        assert torch.autograd.gradcheck(f2, (logp.detach().clone().requires_grad_(True),), eps=1e-9, atol=1e-6)


def test_gspo_gradient_equals_paper_eq10_when_unclipped():
    logp, mask, adv, lens = batch()
    beh = logp.detach() + 0.05 * torch.randn_like(logp)
    lp = logp.detach().clone().requires_grad_(True)
    ls, s, _ = gspo_loss_sum(lp, beh, adv, mask, 10.0, 10.0)
    ls.backward()
    ref = -(s.detach() * adv / lens.to(mask.dtype)).unsqueeze(1) * mask
    assert torch.allclose(lp.grad, ref, atol=1e-14)


def test_fp32_matches_fp64():
    logp, mask, adv, _ = batch()
    beh = logp.detach() + 0.01 * torch.randn_like(logp)
    for fn in (grpo_len_loss_sum, gspo_loss_sum):
        g = {}
        for dt in (torch.float64, torch.float32):
            lp = logp.detach().to(dt).clone().requires_grad_(True)
            ls, _, _ = fn(lp, beh.to(dt), adv.to(dt), mask.to(dt), 0.2, 0.2)
            ls.backward()
            g[dt] = lp.grad.double()
        assert torch.allclose(g[torch.float32], g[torch.float64], atol=1e-6)


def test_legacy_grpo_unchanged():
    # grpo_loss_sum (constant normalization) is the reference objective and stays as it was.
    logp = torch.tensor([[-1.0, -2.0], [-0.5, -0.1]], requires_grad=True)
    ls, ratio = grpo_loss_sum(logp, logp.detach(), torch.tensor([1.0, -1.0]), torch.ones(2, 2), 0.2)
    assert torch.equal(ratio, torch.ones(2, 2)) and float(ls) == 0.0


def test_calibration_quantile_and_rounding(tmp_path):
    import subprocess
    import sys
    import json as _j
    import os
    import numpy as np
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rng = np.random.default_rng(0)
    for init in ("light", "extended"):
        for obj, unit, scale in (("grpolen", "tok", 0.05), ("gspo", "seq", 0.001)):
            d = tmp_path / f"bc2rl-gp-{init}-{obj}-s11" / "ratios"
            d.mkdir(parents=True)
            for r in (1, 2):
                z = {f"{unit}1": np.zeros(50, np.float32)}            # step 1 is excluded (ratio exactly 1)
                for k in (2, 3, 4):
                    z[f"{unit}{k}"] = rng.uniform(-scale, scale, 1000).astype(np.float32)
                np.savez(d / f"round{r:04d}.npz", **z)
        d = tmp_path / f"bc2rl-gp-{init}-ctpo-s11" / "ratios"
        d.mkdir(parents=True)
        for r in (1, 2):
            z = {"cum1": np.zeros(50, np.float32), "pos1": np.ones(50, np.int16)}
            for k in (2, 3, 4):
                z[f"cum{k}"] = rng.uniform(-0.1, 0.1, 4000).astype(np.float32)
                z[f"pos{k}"] = np.ones(4000, np.int16)                  # t = 1: adaptive and fixed bands coincide
            np.savez(d / f"round{r:04d}.npz", **z)
    subprocess.run([sys.executable, "scripts/g_calibrate.py", "--runs", str(tmp_path), "--out", str(tmp_path / "o")],
                   cwd=root, check=True, env=dict(os.environ, PYTHONPATH=root))
    res = _j.load(open(tmp_path / "o" / "g_calibration.json"))
    # |expm1(U(-a, a))| ~ U(0, a) -> 85th percentile ~ 0.85 a
    assert abs(res["eps_tok_raw"] - 0.85 * 0.05) < 0.003 and abs(res["eps_seq_raw"] - 0.85 * 0.001) < 6e-5
    assert res["n_tok"] == 2 * 2 * 3 * 1000 and res["eps_seq"] == float(f"{res['eps_seq_raw']:.2g}")
    # U(-0.1, 0.1) with band (-e, 2e): P(x < -e) = 0.15 and 2e > 0.1 -> e = 0.07
    for name in ("ctpo", "ctpo_fixed"):
        assert abs(res[name]["clip_low"] - 0.07) <= 0.003 and res[name]["clip_c"] == res[name]["clip_high"]


# ----------------------------------------------------------------------------- CTPO (A6 Section 12.8)
from bc2rl.losses import ctpo_loss_sum  # noqa: E402


def ref_ctpo_pg(old_log_prob, log_prob, advantages, response_mask, lo, hi, c, p):
    """Transcription of horizon-llm/CTPO `_compute_policy_loss_cum_token_la_impl` (ratio_mode cumprod) up to the
    per-token loss matrix; advantages are per token (B, T) as in verl."""
    nk = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    mask = response_mask.to(dtype=nk.dtype)
    cs = torch.cumsum(nk * mask, dim=-1)
    kl_values = torch.where(mask > 0, cs, torch.zeros_like(cs))
    lir = torch.clamp(kl_values.detach() + log_prob - log_prob.detach(), max=10.0)
    ratio = torch.where(mask > 0, torch.exp(lir), torch.zeros_like(lir))
    pos_t = torch.cumsum(mask, dim=-1)
    scale = pos_t.clamp(min=1.0).pow(p)
    alo, ahi, ac = lo * scale, hi * scale, c * scale
    lrc = torch.clamp(lir, -alo, ahi)
    cr = torch.where(mask > 0, torch.exp(lrc), torch.zeros_like(lrc))
    dct = torch.where(mask > 0, torch.exp(ac), torch.ones_like(ac))
    l1, l2 = -advantages * ratio, -advantages * cr
    c1 = torch.maximum(l1, l2)
    c2 = torch.min(-advantages * dct, c1)
    return torch.where(advantages < 0, c2, c1)


def test_ctpo_matches_reference_values_and_gradients():
    torch.manual_seed(1)
    logp, mask, adv, lens = batch()
    beh = logp.detach() + 0.08 * torch.randn_like(logp)          # many tokens clipped at these bounds
    for p in (0.5, 0.0):
        lp1 = logp.detach().clone().requires_grad_(True)
        ls, _, _ = ctpo_loss_sum(lp1, beh, adv, mask, 0.025, 0.05, 0.05, p)
        ls.backward()
        lp2 = logp.detach().clone().requires_grad_(True)
        pg = ref_ctpo_pg(beh, lp2, adv.unsqueeze(1).expand_as(lp2), mask, 0.025, 0.05, 0.05, p)
        ref = ((pg * mask).sum(1) / mask.sum(1)).sum()
        ref.backward()
        assert torch.allclose(ls, ref, atol=1e-12) and torch.allclose(lp1.grad, lp2.grad, atol=1e-12)


def test_ctpo_first_update_equals_length_normalized_grpo():
    logp, mask, adv, lens = batch()
    lp = logp.detach().clone().requires_grad_(True)
    ls, lr, st = ctpo_loss_sum(lp, lp.detach(), adv, mask, 0.025, 0.05, 0.05, 0.5)
    assert torch.all(lr == 0) and st["outside"] == 0 and st["clipped"] == 0
    (ls / 8).backward()
    closed = -(adv.unsqueeze(1) * mask / lens.unsqueeze(1).to(mask.dtype)) / 8
    assert torch.allclose(lp.grad, closed, atol=1e-15)


def test_ctpo_small_exact_example_and_no_prefix_gradient():
    # one row, 3 tokens, A = 1, d = (0.1, -0.05, 0.2) -> C = (0.1, 0.05, 0.25); hi_t = 0.05 sqrt(t)
    beh = torch.zeros(1, 3, dtype=torch.float64)
    lp = torch.tensor([[0.1, -0.05, 0.2]], dtype=torch.float64, requires_grad=True)
    mask = torch.ones(1, 3, dtype=torch.float64)
    ls, lr, st = ctpo_loss_sum(lp, beh, torch.tensor([1.0], dtype=torch.float64), mask, 0.025, 0.05, 0.05, 0.5)
    assert torch.allclose(lr, torch.tensor([[0.1, 0.05, 0.25]], dtype=torch.float64))
    exp = -(math.exp(0.05) + math.exp(0.05) + math.exp(0.05 * math.sqrt(3))) / 3
    assert abs(float(ls) - exp) < 1e-12
    ls.backward()
    # tokens 1 and 3 are clipped (zero gradient); token 2 gets -A rho_2 / 3 and nothing from later tokens
    assert lp.grad[0, 0] == 0 and lp.grad[0, 2] == 0
    assert abs(float(lp.grad[0, 1]) + math.exp(0.05) / 3) < 1e-12
    assert st["clipped"] == 2 and st["outside"] == 2
    # naive differentiation of the cumulative product (no stop-gradient) adds prefix terms: it differs
    lp2 = lp.detach().clone().requires_grad_(True)
    naive = -(torch.exp(torch.cumsum(lp2 - beh, dim=1))).mean()
    naive.backward()
    assert abs(float(lp2.grad[0, 0]) - float(-(torch.exp(torch.cumsum(lp.detach(), 1))).sum() / 3)) < 1e-12
    unclipped = lp.detach().clone().requires_grad_(True)
    l_u, _, _ = ctpo_loss_sum(unclipped, beh, torch.tensor([1.0], dtype=torch.float64), mask, 9.0, 9.0, 9.0, 0.5)
    l_u.backward()
    assert not torch.allclose(unclipped.grad, lp2.grad)
    assert torch.allclose(unclipped.grad, -torch.exp(torch.cumsum(lp.detach(), 1)) / 3)


def test_ctpo_dual_clip_and_lower_clip_for_negative_advantage():
    beh = torch.zeros(2, 1, dtype=torch.float64)
    lp = torch.tensor([[0.3], [-0.1]], dtype=torch.float64, requires_grad=True)
    mask = torch.ones(2, 1, dtype=torch.float64)
    adv = torch.tensor([-1.0, -1.0], dtype=torch.float64)
    ls, _, st = ctpo_loss_sum(lp, beh, adv, mask, 0.025, 0.05, 0.05, 0.5)
    ls.backward()
    # row 0: rho = e^0.3 above the dual threshold e^0.05 -> loss capped at e^0.05, zero gradient
    # row 1: rho = e^-0.1 below e^-0.025 -> clipped at e^-0.025, zero gradient
    assert abs(float(ls) - (math.exp(0.05) + math.exp(-0.025))) < 1e-12
    assert torch.all(lp.grad == 0) and st["clipped"] == 2


def test_ctpo_masking_positions_and_stability():
    logp, mask, adv, lens = batch()
    beh = logp.detach() + 0.01 * torch.randn_like(logp)
    lp1 = logp.detach().clone().requires_grad_(True)
    l1, lr1, _ = ctpo_loss_sum(lp1, beh, adv, mask, 0.025, 0.05, 0.05, 0.5)
    l1.backward()
    lp2 = (logp.detach() + (1 - mask) * 50.0).requires_grad_(True)
    l2, lr2, _ = ctpo_loss_sum(lp2, beh, adv, mask, 0.025, 0.05, 0.05, 0.5)
    l2.backward()
    assert torch.allclose(l1, l2) and torch.allclose(lp1.grad, lp2.grad) and torch.all(lp1.grad * (1 - mask) == 0)
    assert torch.all(lr1 * (1 - mask) == 0)
    # huge per-token log-ratios: clamp at +-20 and log rho <= 10 keep everything finite
    lp3 = (beh + 100.0 * mask).requires_grad_(True)
    l3, lr3, _ = ctpo_loss_sum(lp3, beh, adv, mask, 0.025, 0.05, 0.05, 0.5)
    l3.backward()
    assert torch.isfinite(l3) and torch.all(torch.isfinite(lp3.grad)) and float(lr3.max()) <= 10.0
    # unclipped probe settings (infinite band) stay finite, including rows with A = 0
    lp4 = logp.detach().clone().requires_grad_(True)
    inf = float("inf")
    l4, _, st4 = ctpo_loss_sum(lp4, beh, adv, mask, inf, inf, inf, 0.5)
    l4.backward()
    assert torch.isfinite(l4) and torch.all(torch.isfinite(lp4.grad)) and st4["clipped"] == 0


def test_ctpo_gradcheck_fp64():
    logp, mask, adv, _ = batch()
    beh = logp.detach() + 0.03 * torch.randn_like(logp)
    f = lambda x: ctpo_loss_sum(x, beh, adv, mask, 0.025, 0.05, 0.05, 0.5)[0]
    assert torch.autograd.gradcheck(f, (logp.detach().clone().requires_grad_(True),), eps=1e-9, atol=1e-6)
