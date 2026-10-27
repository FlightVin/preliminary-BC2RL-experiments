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
