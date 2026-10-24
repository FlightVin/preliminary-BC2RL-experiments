import torch

from bc2rl.losses import filtered_loss_sum, grpo_advantages, grpo_loss_sum


def test_advantages_population_std():
    r = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    a = grpo_advantages(r, 4)
    # group 1: mean .75, pop std sqrt(.1875)=.4330 ; group 2: mean .25
    s = 0.75 ** 0.5 / 2
    exp = torch.tensor([.25 / s, -.75 / s, .25 / s, .25 / s, -.25 / s, -.25 / s, -.25 / s, .75 / s])
    assert torch.allclose(a, exp / (1 + 1e-6 / s), atol=1e-5)
    assert torch.all(grpo_advantages(torch.ones(4), 4) == 0)
    assert torch.all(grpo_advantages(torch.zeros(4), 4) == 0)


def test_grpo_hand_computed_batch():
    # Q=4 completions, G=2, Lmax=4. Group 1 rewards (1, 0); group 2 (1, 1) -> zero advantage.
    Q, L = 4, 4
    r = torch.tensor([1.0, 0.0, 1.0, 1.0])
    adv = grpo_advantages(r, 2)
    a = 0.5 / (0.5 + 1e-6)
    assert torch.allclose(adv, torch.tensor([a, -a, 0.0, 0.0]))
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.float32)
    logp = torch.full((Q, L), -1.0, requires_grad=True)
    old = torch.full((Q, L), -1.0)
    ls, ratio = grpo_loss_sum(logp, old, adv, mask)
    loss = ls / (Q * L)
    # sum_i sum_t A_i * 1 over masked tokens = a*2 - a*3 = -a  ->  loss = a / 16
    assert abs(float(loss) - a / 16) < 1e-6
    loss.backward()
    exp_grad = -(adv.unsqueeze(1) * mask) / (Q * L)
    assert torch.allclose(logp.grad, exp_grad, atol=1e-7)
    assert torch.allclose(ratio, torch.ones_like(ratio))


def test_grpo_clipping_branches():
    adv = torch.tensor([1.0])
    mask = torch.ones(1, 1)
    hi = torch.tensor([[torch.log(torch.tensor(1.5))]], requires_grad=True)
    ls, _ = grpo_loss_sum(hi, torch.zeros(1, 1), adv, mask)
    assert abs(float(ls) + 1.2) < 1e-6
    ls.backward()
    assert float(hi.grad) == 0.0                     # clipped: no gradient
    lo = torch.tensor([[torch.log(torch.tensor(0.5))]], requires_grad=True)
    ls, _ = grpo_loss_sum(lo, torch.zeros(1, 1), adv, mask)
    assert abs(float(ls) + 0.5) < 1e-6               # min(0.5, 0.8) -> unclipped branch
    ls.backward()
    assert float(lo.grad) != 0.0


def test_filtered_loss_reduction():
    logp = torch.tensor([[-1.0, -2.0, 0.0], [-0.5, 0.0, 0.0]], requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    S, L = 2, 3
    loss = filtered_loss_sum(logp, mask) / (S * L)
    assert abs(float(loss) - 3.5 / 6) < 1e-7
    loss.backward()
    assert torch.allclose(logp.grad, -mask / (S * L))
