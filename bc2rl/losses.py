"""Online objectives (proposal Section 7; PROJECT.md "Online objectives").

GRPO:      A_i = (r_i - mean_g) / (std_g + 1e-6), population std within the prompt group.
           loss = -(1 / (Q * Lmax)) * sum_i sum_t min(rho_it A_i, clip(rho_it, 1-eps, 1+eps) A_i)
Filtered:  loss = -(1 / (|S| * Lmax)) * sum_{i in S} sum_t log p(y_it)
Masks keep completion tokens only. Q is the rollout batch size (128 in the main matrix).
"""
from __future__ import annotations

import torch


def grpo_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-6) -> torch.Tensor:
    """rewards: (Q,) ordered so each consecutive block of `group_size` is one prompt group."""
    r = rewards.float().view(-1, group_size)
    mean = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, unbiased=False, keepdim=True)
    return ((r - mean) / (std + eps)).view(-1)


def grpo_loss_sum(logp, old_logp, adv, mask, clip_eps: float = 0.2):
    """Un-normalised clipped surrogate summed over masked tokens (negated, to minimise).

    logp, old_logp, mask: (B, T); adv: (B,). Divide the sum by Q * Lmax outside, so that
    micro-batch accumulation reproduces the full-batch loss exactly.
    Returns (loss_sum, ratio tensor).
    """
    ratio = torch.exp(logp - old_logp)
    a = adv.unsqueeze(1).to(logp.dtype)
    s1 = ratio * a
    s2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * a
    obj = torch.minimum(s1, s2) * mask
    return -obj.sum(), ratio


def filtered_loss_sum(logp, mask):
    """Negated log-likelihood summed over masked tokens of the selected (successful) rows."""
    return -(logp * mask).sum()


def token_logprobs(logits: torch.Tensor, targets: torch.Tensor, want_entropy: bool = True):
    """Per-token log p(target) and (optionally) entropy of the next-token distribution (float32)."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    tok = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    ent = -(lp.exp() * lp).sum(-1) if want_entropy else None
    return tok, ent
