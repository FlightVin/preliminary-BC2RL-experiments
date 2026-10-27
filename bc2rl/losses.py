"""Online objectives (proposal Section 7; PROJECT.md "Online objectives").

GRPO:      A_i = (r_i - mean_g) / (std_g + 1e-6), population std within the prompt group.
           loss = -(1 / (Q * Lmax)) * sum_i sum_t min(rho_it A_i, clip(rho_it, 1-eps, 1+eps) A_i)
Filtered:  loss = -(1 / (|S| * Lmax)) * sum_{i in S} sum_t log p(y_it)
Masks keep completion tokens only. Q is the rollout batch size (128 in the main matrix).

A6 Section 12 (GSPO stage; `online.objective`, absent = the constant-normalization GRPO above):
grpo_len:  loss = -(1 / Q) * sum_i (1 / |y_i|) * sum_t min(r_it A_i, clip(r_it, 1-e_lo, 1+e_hi) A_i)
gspo:      loss = -(1 / Q) * sum_i min(s_i A_i, clip(s_i, 1-e_lo, 1+e_hi) A_i),
           s_i = exp((1 / |y_i|) * sum_t (log p(y_it) - log b(y_it)))      (GSPO, arXiv 2507.18071, Eq. 5-7)
|y_i| = completion tokens including the stop token (truncated rows: the cap). b = frozen behaviour log-probs.
ctpo:      CTPO (arXiv 2605.07331; reference `cum-token-cumprod-la`, horizon-llm/CTPO verl/trainer/ppo/core_algos.py):
           d_t = clamp(log p - log b, -20, 20); C_t = sum_{t' <= t} d_t'; log rho_t = sg[C_t] + log p_t - sg[log p_t]
           (value C_t, gradient only through the current token: no prefix-gradient terms), clamped at <= 10;
           band in log space [-e_lo t^pw, e_hi t^pw], t = 1-based completion position; dual clip for A < 0 at
           exp(e_c t^pw). Per-token surrogate as the reference; aggregation here is per-sequence mean then / Q
           (the paper's formula; the reference code defaults to a global token mean).
With b = logp.detach() (first update of a batch) all three give the gradient -(1/Q) sum_i A_i / |y_i| sum_t grad log p.
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


def _clip_stats(ratio, adv, unit_mask, e_lo, e_hi):
    """Counts over the units in `unit_mask`: outside the band, and clipped in the objective (zero gradient)."""
    out = ((ratio < 1.0 - e_lo) | (ratio > 1.0 + e_hi)).float() * unit_mask
    act = (((adv > 0) & (ratio > 1.0 + e_hi)) | ((adv < 0) & (ratio < 1.0 - e_lo))).float() * unit_mask
    return float(out.sum()), float(act.sum())


def grpo_len_loss_sum(logp, beh_logp, adv, mask, e_lo: float, e_hi: float):
    """Token-ratio clipped surrogate with per-sequence length normalization, summed over rows (negated).

    logp, beh_logp, mask: (B, T); adv: (B,). Divide by Q outside. Returns (loss_sum, ratio, stats).
    stats: token counts outside the band / clipped, over masked tokens of rows with adv != 0.
    """
    n = mask.sum(1).clamp(min=1.0)
    ratio = torch.exp(logp - beh_logp)
    a = adv.unsqueeze(1).to(logp.dtype)
    obj = torch.minimum(ratio * a, torch.clamp(ratio, 1.0 - e_lo, 1.0 + e_hi) * a) * mask
    unit = mask * (adv != 0).unsqueeze(1).to(mask.dtype)
    o, c = _clip_stats(ratio.detach(), a.expand_as(ratio), unit, e_lo, e_hi)
    return -(obj.sum(1) / n).sum(), ratio, {"units": float(unit.sum()), "outside": o, "clipped": c}


def gspo_loss_sum(logp, beh_logp, adv, mask, e_lo: float, e_hi: float):
    """GSPO sequence-ratio clipped surrogate summed over rows (negated). Divide by Q outside.

    Returns (loss_sum, s (B,), stats). stats count sequences (rows with adv != 0 and >= 1 token) and the
    tokens of clipped sequences.
    """
    n = mask.sum(1).clamp(min=1.0)
    log_s = ((logp - beh_logp) * mask).sum(1) / n
    s = torch.exp(log_s)
    a = adv.to(logp.dtype)
    obj = torch.minimum(s * a, torch.clamp(s, 1.0 - e_lo, 1.0 + e_hi) * a)
    unit = ((adv != 0) & (mask.sum(1) > 0)).to(mask.dtype)
    o, c = _clip_stats(s.detach(), a, unit, e_lo, e_hi)
    act = (((a > 0) & (s.detach() > 1.0 + e_hi)) | ((a < 0) & (s.detach() < 1.0 - e_lo))).to(mask.dtype) * unit
    return -obj.sum(), s, {"units": float(unit.sum()), "outside": o, "clipped": c,
                           "clipped_tokens": float((act * mask.sum(1)).sum()), "unit_tokens": float((unit * mask.sum(1)).sum())}


CTPO_BUCKETS = ((1, 8), (9, 32), (33, 64), (65, 128), (129, 256))


def ctpo_loss_sum(logp, beh_logp, adv, mask, e_lo: float, e_hi: float, e_c: float, power: float):
    """CTPO-LA surrogate loss (reference semantics), per-sequence mean, summed over rows. Divide by Q outside.

    Returns (loss_sum, log_rho (B, T) detached, stats). stats: token units (rows with adv != 0), tokens outside the
    band, tokens clipped (zero gradient, incl. the dual clip), and per-position-bucket outside / clipped counts.
    """
    m = mask.to(logp.dtype)
    d = torch.clamp(logp - beh_logp, min=-20.0, max=20.0)
    cum = torch.where(m > 0, torch.cumsum(d * m, dim=-1), torch.zeros_like(d))
    log_rho = torch.clamp(cum.detach() + logp - logp.detach(), max=10.0)
    rho = torch.where(m > 0, torch.exp(log_rho), torch.zeros_like(log_rho))
    pos = torch.cumsum(m, dim=-1)
    scale = pos.clamp(min=1.0).pow(power) if power != 0 else torch.ones_like(pos)
    lo, hi, c = e_lo * scale, e_hi * scale, e_c * scale
    lr_clip = torch.minimum(torch.maximum(log_rho, -lo), hi)
    rho_clip = torch.where(m > 0, torch.exp(lr_clip), torch.zeros_like(lr_clip))
    dual = torch.where(m > 0, torch.exp(c), torch.ones_like(c))
    a = adv.unsqueeze(1).to(logp.dtype).expand_as(logp)
    l1, l2 = -a * rho, -a * rho_clip
    cl1 = torch.maximum(l1, l2)
    cl2 = torch.minimum(-a * dual, cl1)
    pg = torch.where(a < 0, cl2, cl1)
    n = m.sum(1).clamp(min=1.0)
    loss = ((pg * m).sum(1) / n).sum()
    with torch.no_grad():
        lr = log_rho.detach()
        unit = m * (adv != 0).unsqueeze(1).to(m.dtype)
        outside = ((lr < -lo) | (lr > hi)).to(m.dtype) * unit
        clipped = ((((a > 0) & (lr > hi)) | ((a < 0) & (lr < -lo)) | ((a < 0) & (cl1 > -a * dual)))
                   .to(m.dtype) * unit)
        st = {"units": float(unit.sum()), "outside": float(outside.sum()), "clipped": float(clipped.sum())}
        for b0, b1 in CTPO_BUCKETS:
            inb = ((pos >= b0) & (pos <= b1)).to(m.dtype) * unit
            st[f"u{b0}"], st[f"o{b0}"], st[f"c{b0}"] = float(inb.sum()), float((outside * inb).sum()), float((clipped * inb).sum())
    return loss, lr, st          # pg terms are already losses (-A rho), as in the reference


def filtered_loss_sum(logp, mask):
    """Negated log-likelihood summed over masked tokens of the selected (successful) rows."""
    return -(logp * mask).sum()


def token_logprobs(logits: torch.Tensor, targets: torch.Tensor, want_entropy: bool = True):
    """Per-token log p(target) and (optionally) entropy of the next-token distribution (float32)."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    tok = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    ent = -(lp.exp() * lp).sum(-1) if want_entropy else None
    return tok, ent
