"""E2 matched-update intervention (`bc2rl mechdata`, `bc2rl mech`). Protocol: protocol/AMENDMENT-A2 Section 4.

Stages of `bc2rl mech` (each saved atomically; a resumed job skips finished stages):
  1. pool     sample G completions for each source prompt (blocks of 128 prompts), exact verification
  2. select   per-group GRPO gradients (bf16 storage, search only), anchor-fit direction h, pools,
              exact (input count, m) matching, length rule, high/low local search on actual Adam displacements
  3. disp     exact displacements: fp32 backward over each arm's 32 completions, fresh-state AdamW step,
              norm equalization within the pair
  4. forks    each displacement x rho applied once to an exact copy of the checkpoint (restore verified);
              anchor-fit / anchor-check log-likelihood (token classes), probe pass counts (15 attempts),
              sampled-token KL, Taylor check
mode "calibrate" (development only) replaces stage 4 with the radius grid and writes the frozen rho rule.

Amendment A5 / X6 (protocol/AMENDMENT-A5 Section 4), off by default so A2 configurations behave as before:
  fp32_measure: true   h, every log-likelihood and every Taylor quantity use an fp32 copy of the model; sampling,
                       probes and the KL statistic stay on the BF16 policy (adapter weights are fp32 in both).
  mode "calibrate2"    development radius grid below one step with validity checks V1-V5 (anchor-fit + dev held-out,
                       random sign directions); writes calibration.json; `scripts/x6_gate.py` combines checkpoints.
  validity_check: true per-checkpoint V1-V3 on anchor-fit before the forks of a study run; a failing checkpoint is
                       recorded as invalid and gets no forks.
"""
from __future__ import annotations

import copy

import gzip
import hashlib
import json
import os
import random
import time
from collections import Counter

import numpy as np
import torch

from . import task
from .data import data_dir, load_data
from .evaluate import counts_by_problem, run_attempts, stable_int
from .losses import grpo_advantages
from .modeling import (chat_prompt, gather_completion, load_adapter, load_policy, load_tokenizer, model_info,
                       sample, seq_logprobs)
from .runtime import CostLedger, Run, StopFlag, atomic_json, read_jsonl_gz, write_jsonl_gz

MECH_SPLITS = ("anchorfit", "anchorcheck", "probe")


# ----------------------------------------------------------------------------- data extension

def mech_dir(cfg):
    return os.path.join(data_dir(cfg), "mech")


def cmd_mechdata(cfg):
    """Draw anchor-fit / anchor-check / probe problems with the frozen generator config and a new seed,
    deduplicated against every existing split by canonical key. Never modifies the v1 manifest."""
    d = mech_dir(cfg)
    mpath = os.path.join(d, "mech_manifest.json")
    if os.path.exists(mpath):
        load_mech(cfg)
        print(f"mech data {d} exists; hashes OK")
        return 0
    base = os.path.join(data_dir(cfg), "manifest.json")
    man = json.load(open(base))
    seen = set()
    for s in list(man["splits"]) + ["format"]:
        for p in task.load_split(os.path.join(data_dir(cfg), f"{s}.jsonl")):
            seen.add(p.key)
    n_existing = len(seen)
    g = cfg["data"]
    gcfg = task.GenConfig(min_value=g["min_value"], max_value=g["max_value"], max_target=g["max_target"],
                          max_abs_intermediate=g["max_abs_intermediate"], ops=tuple(g["ops"]))
    md = cfg["mechdata"]
    rng = random.Random(md["seed"])
    out, stats = {}, {}
    for split in MECH_SPLITS:
        out[split] = []
        for h in g["horizons"]:
            st = stats.setdefault(f"{split}:{h}", Counter())
            got = []
            while len(got) < md["per_inputs"]:
                values, target, trace, structure, ops_used = task.sample_problem(rng, h, gcfg, st)
                key = task.canonical_key(values, target)
                if key in seen:
                    st["reject:duplicate"] += 1
                    continue
                seen.add(key)
                shown = values[:]
                rng.shuffle(shown)
                got.append(task.Problem(pid=f"{split}-{h}-{len(got):05d}", split=split, n_inputs=h, inputs=shown,
                                        target=target, ref_trace=trace, structure=structure, ops_used=ops_used))
            out[split] += got
    os.makedirs(d, exist_ok=True)
    for s, probs in out.items():
        task.save_split(os.path.join(d, f"{s}.jsonl"), probs)
    mman = {"base_manifest_sha256": hashlib.sha256(open(base, "rb").read()).hexdigest(),
            "generator": task.GENERATOR_VERSION, "gen_config": g, "seed": md["seed"],
            "existing_keys_excluded": n_existing, "rejection_stats": {k: dict(v) for k, v in stats.items()},
            "splits": {s: {"count": len(p), "by_inputs": dict(Counter(x.n_inputs for x in p)),
                           "sha256": task.manifest_hash(p)} for s, p in out.items()}}
    atomic_json(mpath, mman)
    if cfg.get("_results_dir"):
        atomic_json(os.path.join(cfg["_results_dir"], "mech_manifest.json"), mman)
    print(json.dumps(mman["splits"]))
    return 0


def load_mech(cfg):
    d = mech_dir(cfg)
    man = json.load(open(os.path.join(d, "mech_manifest.json")))
    out = {}
    for s in MECH_SPLITS:
        probs = task.load_split(os.path.join(d, f"{s}.jsonl"))
        if task.manifest_hash(probs) != man["splits"][s]["sha256"]:
            raise SystemExit(f"mech split {s} hash mismatch")
        out[s] = probs
    return out, man


# ----------------------------------------------------------------------------- vector helpers

def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def flat_params(params):
    return torch.cat([p.detach().reshape(-1) for p in params])


def set_flat(params, vec):
    i = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            p.copy_(vec[i:i + n].view_as(p))
            i += n


def flat_grad(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])


def vec_sha(vec):
    return hashlib.sha256(vec.detach().float().cpu().numpy().tobytes()).hexdigest()


def clip_like_torch(g, max_norm=1.0):
    """torch.nn.utils.clip_grad_norm_: coef = max_norm / (norm + 1e-6), applied only when coef < 1."""
    n = g.norm()
    coef = torch.clamp(max_norm / (n + 1e-6), max=1.0)
    return g * coef


def adam_first_step(g, lr, eps=1e-8, max_norm=1.0):
    """Parameter change of one AdamW step from a fresh state (beta1 0.9, beta2 0.999, weight decay 0) on the
    LOSS gradient g after norm clipping: d = -lr * g / (|g| + eps) (bias corrections cancel at step 1)."""
    gc = clip_like_torch(g.float(), max_norm)
    return -lr * gc / (gc.abs() + eps)


def cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))


# ----------------------------------------------------------------------------- log-likelihood pieces

def encode_ref(tok, problem, fmt, variant):
    p_ids = tok(chat_prompt(tok, task.prompt_text(problem, variant, fmt)), add_special_tokens=False).input_ids
    c_ids = tok(task.completion_text(problem), add_special_tokens=False).input_ids + \
        [tok.convert_tokens_to_ids("<|im_end|>")]
    return p_ids, c_ids


def token_classes(tok, c_ids):
    """1 = numeric/operator token (contains a digit or + - *), 0 = delimiter (space, '=', newline, end-of-turn)."""
    out = []
    for t in c_ids:
        s = tok.decode([t])
        out.append(1 if any(ch.isdigit() or ch in "+-*" for ch in s) else 0)
    return out


def anchor_ll(model, seqs, classes, pad_id, Lmax, micro):
    """Mean over problems of (sum completion log p)/Lmax, plus per-problem values and token-class sums."""
    per, num, dlm, n_num, n_dlm = [], 0.0, 0.0, 0, 0
    for sl, lp, cm, _ in seq_logprobs(model, seqs, pad_id, micro, grad=False):
        for r in range(lp.shape[0]):
            k = sl.start + r
            row = lp[r][cm[r].bool()].float()             # all completion tokens, in order
            assert row.shape[0] == len(seqs[k][1])
            per.append(float(row.sum()) / Lmax)
            cl = torch.tensor(classes[k], device=row.device, dtype=torch.bool)
            num += float(row[cl].sum())
            dlm += float(row[~cl].sum())
            n_num += int(cl.sum())
            n_dlm += int((~cl).sum())
    return {"mean_norm_ll": float(np.mean(per)), "per_problem": per,
            "num_ll_per_token": num / max(n_num, 1), "delim_ll_per_token": dlm / max(n_dlm, 1),
            "n_num_tokens": n_num, "n_delim_tokens": n_dlm}


def anchor_direction(model, params, seqs, pad_id, Lmax, micro):
    """h = gradient of the mean over problems of (sum completion log p)/Lmax (ascent direction)."""
    model.zero_grad(set_to_none=True)
    N = len(seqs)
    for sl, lp, cm, _ in seq_logprobs(model, seqs, pad_id, micro, grad=True):
        loss = (lp * cm).sum() / (N * Lmax)
        loss.backward()
    h = flat_grad(params).clone()
    model.zero_grad(set_to_none=True)
    return h


def grpo_loss_grad(model, params, seqs, adv, pad_id, Lmax, micro, Qloc):
    """Loss gradient of -(1/(Qloc*Lmax)) sum_i A_i sum_t log p (ratio 1: same-pass surrogate). fp32 vector."""
    model.zero_grad(set_to_none=True)
    for sl, lp, cm, _ in seq_logprobs(model, seqs, pad_id, micro, grad=True):
        a = adv[sl.start:sl.stop].to(lp.device).unsqueeze(1)
        loss = -(lp * cm * a).sum() / (Qloc * Lmax)
        loss.backward()
    g = flat_grad(params).clone()
    model.zero_grad(set_to_none=True)
    return g


# ----------------------------------------------------------------------------- pools, matching, search

def make_pools(keys, max_pools=8, per_pool=16, seed=777):
    """keys: list of (n_inputs, m) per mixed group (index = group id). Shuffle within key (seeded), deal
    round-robin over P = min(max_pools, len // per_pool) pools. Returns list of lists of group ids."""
    P = min(max_pools, len(keys) // per_pool)
    if P == 0:
        return []
    rng = np.random.default_rng(seed)
    by = {}
    for i, k in enumerate(keys):
        by.setdefault(tuple(k), []).append(i)
    pools = [[] for _ in range(P)]
    c = 0
    for k in sorted(by):
        ids = [by[k][j] for j in rng.permutation(len(by[k]))]
        for i in ids:
            pools[c % P].append(i)
            c += 1
    return pools


def apportion(counts, total=8):
    """counts: {key: n in pool}. Per-arm count c_k: largest remainder of `total` in proportion to n_k,
    with c_k <= n_k // 2. Returns {key: c_k} or None when infeasible."""
    caps = {k: n // 2 for k, n in counts.items()}
    if sum(caps.values()) < total:
        return None
    S = sum(counts.values())
    q = {k: total * n / S for k, n in counts.items()}
    c = {k: min(int(np.floor(q[k])), caps[k]) for k in counts}
    order = sorted(counts, key=lambda k: (-(q[k] - np.floor(q[k])), -counts[k], k))
    while sum(c.values()) < total:
        progressed = False
        for k in order:
            if sum(c.values()) >= total:
                break
            if c[k] < caps[k]:
                c[k] += 1
                progressed = True
        if not progressed:
            return None
    return {k: v for k, v in c.items() if v > 0}


class Searcher:
    """High/low arm search inside one pool on actual fresh-state Adam displacements of summed group gradients."""

    def __init__(self, G, h, lens, lr, len_tol=0.05, budget=3000):
        self.G, self.h, self.lens, self.lr, self.tol, self.budget = G, h, lens, lr, len_tol, budget
        self.evals = 0

    def align(self, ids):
        g = self.G[list(ids)].float().sum(0)
        self.evals += 1
        return cos(self.h, adam_first_step(g, self.lr))

    def len_viol(self, H, L):
        a, b = float(np.mean([self.lens[i] for i in H])), float(np.mean([self.lens[i] for i in L]))
        return abs(a - b) / max(min(a, b), 1e-9)

    def score(self, H, L):
        v = self.len_viol(H, L)
        return (max(0.0, v - self.tol), -(self.align(H) - self.align(L)))

    def run(self, pool, keys, comp, single):
        """pool: group ids; keys: key per id; comp: {key: c}; single: id -> single-group alignment."""
        byk = {}
        for i in pool:
            byk.setdefault(tuple(keys[i]), []).append(i)
        H, L = [], []
        for k, ck in sorted(comp.items()):
            ranked = sorted(byk[k], key=lambda i: (-single[i], i))
            H += ranked[:ck]
            L += ranked[-ck:]
        best = self.score(H, L)

        def neighbours(H, L):
            for arm in ("H", "L"):
                cur, oth = (H, L) if arm == "H" else (L, H)
                for k in sorted(comp):
                    for x in [i for i in cur if tuple(keys[i]) == k]:
                        for y in byk[k]:
                            if y in cur:
                                continue
                            nc = [y if i == x else i for i in cur]
                            no = [x if i == y else i for i in oth] if y in oth else list(oth)
                            yield (nc, no) if arm == "H" else (no, nc)

        while self.evals < self.budget:            # first improvement, then rescan from the new state
            improved = False
            for nH, nL in neighbours(H, L):
                if self.evals >= self.budget:
                    break
                s = self.score(nH, nL)
                if s < best:
                    best, H, L, improved = s, nH, nL, True
                    break
            if not improved:
                break
        return H, L, {"len_violation": self.len_viol(H, L), "search_align_H": self.align(H),
                      "search_align_L": self.align(L), "evals": self.evals}


# ----------------------------------------------------------------------------- KL and evaluation

def sampled_kl(model, params, theta0, delta, seqs, pad_id, micro):
    """Per sequence: mean over completion tokens of KL(p_theta0 || p_theta0+delta) over the full vocabulary,
    at contexts sampled from p_theta0. Restores theta0 exactly. Returns a list of per-sequence values."""
    out = []
    from .modeling import build_batch
    for s in range(0, len(seqs), micro):
        part = seqs[s:s + micro]
        ids, att, cm = build_batch(part, pad_id, model.device)
        sel = cm.bool()
        counts = sel.sum(1).tolist()
        with torch.no_grad():
            set_flat(params, theta0)
            lp0 = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits[:, :-1][sel].float(), -1)
            set_flat(params, theta0 + delta)
            lp1 = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits[:, :-1][sel].float(), -1)
            kl = (lp0.exp() * (lp0 - lp1)).sum(-1)
            for chunk in torch.split(kl, counts):
                out.append(float(chunk.mean()) if chunk.numel() else 0.0)
            del lp0, lp1, kl
    set_flat(params, theta0)
    return out


def probe_counts(recs):
    pids, n, c, h = counts_by_problem(recs)
    errs = Counter(r["error"] for r in recs)
    return {"pids": pids, "n": n.tolist(), "c": c.tolist(), "h": h.tolist(),
            "format_valid": float(np.mean([r["format_valid"] for r in recs])),
            "truncated": float(np.mean([r["finish"] == "length" for r in recs])),
            "errors": dict(errs), "mean_tokens": float(np.mean([r["n_tokens"] for r in recs]))}


# ----------------------------------------------------------------------------- main command

def cmd_mech(cfg):
    run = Run(cfg)
    if run.done_ok():
        print(f"[mech] {run.id} already complete")
        return 0
    stop = StopFlag()
    me = cfg["mech"]
    pd = cfg["proj_data"]
    device = cfg.get("device", "cuda")
    calibrate = me["mode"] == "calibrate"
    calibrate2 = me["mode"] == "calibrate2"
    splits = sorted(set(me["source_splits"] + ([me["kl_split"]] if calibrate or calibrate2 else [])
                        + ([me["heldout_split"]] if calibrate2 else [])))
    data, _ = load_data(cfg, splits)
    fmt = data["format"][0]
    mdata, mman = load_mech(cfg)
    info = model_info(pd)
    tok = load_tokenizer(info["path"])
    model = load_policy(cfg.get("model_path_override") or info["path"], cfg["model"]["lora"], seed=0, device=device)
    apath = me["adapter"] if os.path.isabs(me["adapter"]) else os.path.join(pd, me["adapter"])
    asha = load_adapter(model, apath)
    if me.get("adapter_sha256") and me["adapter_sha256"] != asha:
        raise SystemExit(f"adapter hash {asha} != expected {me['adapter_sha256']}")
    model.eval()                          # dropout is 0; eval mode for every pass
    params = trainable(model)
    theta0 = flat_params(params).clone()
    theta0_sha = vec_sha(theta0)
    if me.get("fp32_measure"):
        meas = copy.deepcopy(model).float()
        meas.eval()
        mparams = trainable(meas)
        if not torch.equal(flat_params(mparams), theta0):
            raise SystemExit("fp32 measurement copy differs from the checkpoint")
    else:
        meas, mparams = model, params
    ledger = CostLedger(run.path("cost.json"), "mech")
    G, cap, Lmax, micro, lr = me["group"], me["cap"], me["cap"], me["micro"], me["lr"]
    Qloc = me["groups_per_arm"] * G
    variant = cfg["prompt_variant"]
    t_start = time.time()
    res_dir = cfg.get("_results_dir")

    # ---- stage 1: source pool
    pool_probs = []
    for s in me["source_splits"]:
        pool_probs += data[s]
    perm = np.random.default_rng(me["pool_seed"]).permutation(len(pool_probs))[:me["pool_prompts"]]
    pool_probs = [pool_probs[i] for i in perm]
    texts = [chat_prompt(tok, task.prompt_text(p, variant, fmt)) for p in pool_probs]
    B = 128
    for b in range(0, len(pool_probs), B):
        bp = run.path("pool", f"block{b // B:03d}.jsonl.gz")
        if os.path.exists(bp):
            continue
        outs = sample(model, tok, texts[b:b + B], G, cap, seed=me["sample_seed"] + b // B, batch_seqs=B * G)
        rows = []
        for o in outs:
            p = pool_probs[b + o["prompt_index"]]
            v = task.verify(p.inputs, p.target, o["text"], truncated=(o["finish"] == "length"))
            if cfg.get("test_mode") and me.get("debug_reward") == "random":
                v["reward"] = int(random.Random(f"{b}:{o['prompt_index']}:{o['attempt']}").random() < 0.5)
            rows.append({"pidx": b + o["prompt_index"], "pid": p.pid, "n_inputs": p.n_inputs, "attempt": o["attempt"],
                         "prompt_ids": o["prompt_ids"], "completion_ids": o["completion_ids"], "text": o["text"],
                         "finish": o["finish"], "n_tokens": o["n_tokens"], "reward": v["reward"], "error": v["error"]})
        write_jsonl_gz(bp, rows)
        ledger.flush(note=f"pool block {b // B}")
        if stop.stop:
            raise SystemExit(143)
    rows = []
    for b in range(0, len(pool_probs), B):
        rows += read_jsonl_gz(run.path("pool", f"block{b // B:03d}.jsonl.gz"))
    rows.sort(key=lambda r: (r["pidx"], r["attempt"]))
    groups = [rows[i:i + G] for i in range(0, len(rows), G)]
    m_all = [sum(r["reward"] for r in g) for g in groups]
    pool_stats = {"prompts": len(groups), "completions": len(rows),
                  "success_rate_by_inputs": {str(h): float(np.mean([r["reward"] for r in rows if r["n_inputs"] == h]))
                                             for h in sorted({r["n_inputs"] for r in rows})},
                  "m_hist_by_inputs": {str(h): dict(Counter(int(m) for g, m in zip(groups, m_all) if g[0]["n_inputs"] == h))
                                       for h in sorted({r["n_inputs"] for r in rows})},
                  "truncated": float(np.mean([r["finish"] == "length" for r in rows]))}
    mixed = [i for i, m in enumerate(m_all) if 0 < m < G][:me["max_mixed"]]
    pool_stats["mixed_groups"] = sum(1 for m in m_all if 0 < m < G)
    pool_stats["mixed_kept"] = len(mixed)

    # anchor sets
    def enc(probs):
        seqs = [encode_ref(tok, p, fmt, variant) for p in probs]
        return seqs, [token_classes(tok, c) for _, c in seqs]
    fit_seqs, fit_cls = enc(mdata["anchorfit"])
    chk_seqs, chk_cls = enc(mdata["anchorcheck"])
    h = anchor_direction(meas, mparams, fit_seqs, tok.pad_token_id, Lmax, micro)
    assert torch.equal(flat_params(params), theta0) and torch.equal(flat_params(mparams), theta0)

    # ---- stage 2: selection
    sel_path = run.path("selection.json")
    if not os.path.exists(sel_path):
        gid_keys = [(groups[i][0]["n_inputs"], int(m_all[i])) for i in mixed]
        lens = [float(np.mean([r["n_tokens"] for r in groups[i]])) for i in mixed]
        D = theta0.numel()
        Gs = torch.zeros((len(mixed), D), dtype=torch.bfloat16, device=theta0.device)
        single = {}
        for j, i in enumerate(mixed):
            seqs = [(r["prompt_ids"], r["completion_ids"]) for r in groups[i]]
            adv = grpo_advantages(torch.tensor([float(r["reward"]) for r in groups[i]]), G)
            g = grpo_loss_grad(model, params, seqs, adv, tok.pad_token_id, Lmax, micro, Qloc)
            if not torch.isfinite(g).all():
                raise SystemExit(f"non-finite group gradient for group {i}")
            Gs[j] = g.to(torch.bfloat16)
            single[j] = cos(h, adam_first_step(g, lr))
        pools = make_pools(gid_keys, me["max_pools"], 2 * me["groups_per_arm"], me["pool_split_seed"])
        pairs = []
        for pi, pool in enumerate(pools):
            comp = apportion(Counter(gid_keys[j] for j in pool), me["groups_per_arm"])
            if comp is None:
                pairs.append({"pair": pi, "status": "infeasible composition", "pool_size": len(pool)})
                continue
            srch = Searcher(Gs, h, lens, lr, me["len_tol"], me["search_budget"])
            Hj, Lj, info_s = srch.run(pool, gid_keys, comp, single)
            pairs.append({"pair": pi, "status": "ok" if info_s["len_violation"] <= me["len_tol"] else "length-unmatched",
                          "pool_size": len(pool), "composition": {f"{k[0]}:{k[1]}": v for k, v in comp.items()},
                          "H": [mixed[j] for j in Hj], "L": [mixed[j] for j in Lj],
                          "len_H": float(np.mean([lens[j] for j in Hj])), "len_L": float(np.mean([lens[j] for j in Lj])),
                          "hist_H": sorted(Counter(gid_keys[j] for j in Hj).items()),
                          "hist_L": sorted(Counter(gid_keys[j] for j in Lj).items()), **info_s})
            assert pairs[-1]["hist_H"] == pairs[-1]["hist_L"], "matching failed"
        del Gs
        atomic_json(sel_path, {"pool_stats": pool_stats, "h_norm": float(h.norm()), "theta0_sha": theta0_sha,
                               "adapter_sha256": asha, "n_pools": len(pools), "pairs": pairs,
                               "single_align_summary": {"mean": float(np.mean(list(single.values()))) if single else None,
                                                        "min": float(min(single.values())) if single else None,
                                                        "max": float(max(single.values())) if single else None}})
        ledger.flush(note="selection")
    sel = json.load(open(sel_path))
    use_pairs = [p for p in sel["pairs"] if p["status"] == "ok"][:me["n_pairs"]]

    # ---- stage 3: exact displacements (fp32), norm equalization; saved so a resume reuses identical vectors
    disp_path = run.path("disp.pt")
    if os.path.exists(disp_path):
        saved = torch.load(disp_path, map_location="cpu", weights_only=False)
        use_pairs = saved["pairs"]
        disp = {tuple(k): v.to(theta0.device) for k, v in saved["disp"]}
    else:
        disp = {}
        for p in use_pairs:
            dd = {}
            for arm in ("H", "L"):
                seqs, advs = [], []
                for i in p[arm]:
                    seqs += [(r["prompt_ids"], r["completion_ids"]) for r in groups[i]]
                    advs.append(grpo_advantages(torch.tensor([float(r["reward"]) for r in groups[i]]), G))
                g = grpo_loss_grad(model, params, seqs, torch.cat(advs), tok.pad_token_id, Lmax, micro, Qloc)
                if not torch.isfinite(g).all():
                    raise SystemExit(f"non-finite batch gradient pair {p['pair']} arm {arm}")
                dd[arm] = {"g_norm": float(g.norm()), "d": adam_first_step(g, lr)}
            nH, nL = float(dd["H"]["d"].norm()), float(dd["L"]["d"].norm())
            s = min(nH, nL)
            for arm, n0 in (("H", nH), ("L", nL)):
                d = dd[arm]["d"] * (s / n0)
                disp[(p["pair"], arm)] = d
                p.setdefault("exact", {})[arm] = {"align": cos(h, d), "g_norm": dd[arm]["g_norm"], "d_norm_raw": n0,
                                                 "d_norm_equal": float(d.norm()), "h_dot_d": float(torch.dot(h, d)),
                                                 "frac_coords_below_eps": float((dd[arm]["d"].abs() < 0.5 * lr).float().mean())}
            p["opposite_sign"] = bool(p["exact"]["H"]["align"] > 0 > p["exact"]["L"]["align"])
            p["exact_contrast"] = p["exact"]["H"]["align"] - p["exact"]["L"]["align"]
        from .runtime import atomic_torch_save
        atomic_torch_save(disp_path, {"pairs": use_pairs, "disp": [(list(k), v.cpu()) for k, v in disp.items()]})
        ledger.flush(note="displacements")
    assert torch.equal(flat_params(params), theta0)

    if calibrate2:
        return calibrate_grid(cfg, run, model, meas, tok, params, mparams, theta0, h, disp, use_pairs, data, fmt,
                              fit_seqs, fit_cls, sel, ledger, res_dir, enc)
    if calibrate:
        return calibrate_radius(cfg, run, model, tok, params, theta0, h, disp, use_pairs, data, fmt, fit_seqs,
                                fit_cls, sel, ledger, res_dir)

    # ---- stage 4: forks
    rho, rho_src = resolve_rho(me, pd)
    probe = mdata["probe"]
    kl_texts = [chat_prompt(tok, task.prompt_text(p, variant, fmt)) for p in probe]
    kl_path = run.path("kl_samples.jsonl.gz")
    if not os.path.exists(kl_path):
        o = sample(model, tok, kl_texts, me["kl_samples"], cap, seed=me["sample_seed"] + 99991, batch_seqs=512)
        write_jsonl_gz(kl_path, [{"prompt_ids": x["prompt_ids"], "completion_ids": x["completion_ids"]} for x in o])
    kl_seqs = [(r["prompt_ids"], r["completion_ids"]) for r in read_jsonl_gz(kl_path)]
    forks = [("base", None)] + [(f"p{p['pair']}-{arm}", disp[(p["pair"], arm)]) for p in use_pairs for arm in ("H", "L")]
    validity = None
    if me.get("validity_check"):
        vpath = run.path("validity.json")
        if not os.path.exists(vpath):
            atomic_json(vpath, checkpoint_validity(me, meas, mparams, theta0, h, disp, fit_seqs, fit_cls,
                                                   tok.pad_token_id, rho))
            ledger.flush(note="validity")
        validity = json.load(open(vpath))
        print(f"[mech] validity at rho {rho}: {validity['checks']}", flush=True)
        if not validity["checks"]["all_ok"]:
            forks = []
    base_fit = None
    for tag, d in forks:
        fpath = run.path("forks", f"{tag}.json")
        if os.path.exists(fpath):
            if tag == "base":
                base_fit = json.load(open(fpath))["anchorfit"]["mean_norm_ll"]
            continue
        set_flat(params, theta0 if d is None else theta0 + rho * d)
        if meas is not model:
            set_flat(mparams, theta0 if d is None else theta0 + rho * d)
        rec = {"tag": tag, "rho": rho, "measure": "fp32" if meas is not model else "bf16"}
        rec["anchorfit"] = anchor_ll(meas, fit_seqs, fit_cls, tok.pad_token_id, Lmax, micro)
        rec["anchorcheck"] = anchor_ll(meas, chk_seqs, chk_cls, tok.pad_token_id, Lmax, micro)
        if meas is not model:
            set_flat(mparams, theta0)
        seed = me["eval_seed"] * 1000003 + stable_int(f"{run.id}:{tag}")
        recs, gen_s = run_attempts(model, tok, probe, fmt, variant, me["probe_attempts"], cap, seed,
                                   batch_seqs=me.get("batch_seqs", 480))
        write_jsonl_gz(run.path("forks", f"{tag}.probe.jsonl.gz"), recs)
        rec["probe"] = probe_counts(recs)
        rec["probe_gen_s"] = gen_s
        set_flat(params, theta0)
        if not torch.equal(flat_params(params), theta0):
            raise SystemExit("checkpoint restore failed")
        if d is not None:
            kl = sampled_kl(model, params, theta0, rho * d, kl_seqs, tok.pad_token_id, micro)
            rec["kl_median"], rec["kl_mean"] = float(np.median(kl)), float(np.mean(kl))
            rec["taylor_pred_fit"] = float(torch.dot(h, rho * d))
            rec["taylor_actual_fit"] = rec["anchorfit"]["mean_norm_ll"] - base_fit
            rec["delta_norm"] = float((rho * d).norm())
        else:
            base_fit = rec["anchorfit"]["mean_norm_ll"]
        if vec_sha(flat_params(params)) != theta0_sha:
            raise SystemExit("restored parameters differ from the checkpoint")
        atomic_json(fpath, rec)
        ledger.flush(note=f"fork {tag}")
        print(f"[mech] fork {tag}: fit {rec['anchorfit']['mean_norm_ll']:.5f} check {rec['anchorcheck']['mean_norm_ll']:.5f} "
              f"probe c={sum(rec['probe']['c'])} kl={rec.get('kl_median')}", flush=True)
        if stop.stop:
            raise SystemExit(143)
    summary = {"run_id": run.id, "config_hash": run.chash, "adapter_sha256": asha, "theta0_sha": theta0_sha,
               "rho": rho, "rho_source": rho_src, "selection": sel, "pairs_used": use_pairs, "validity": validity,
               "measure": "fp32" if meas is not model else "bf16",
               "forks": {t: json.load(open(run.path("forks", f"{t}.json"))) for t, _ in forks},
               "anchor_manifest": mman["splits"], "gpu_s_total": ledger.total_gpu_s(), "wall_s": time.time() - t_start,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    for t in summary["forks"].values():
        t["anchorfit"].pop("per_problem", None)
    atomic_json(run.path("summary.json"), summary)
    if res_dir:
        full = {t: json.load(open(run.path("forks", f"{t}.json"))) for t, _ in forks}
        atomic_json(os.path.join(res_dir, "summary.json"), summary)
        atomic_json(os.path.join(res_dir, "forks_full.json"), full)
    atomic_json(run.path("DONE.json"), {"config_hash": run.chash, "artifacts": []})
    ledger.flush(note="complete")
    return 0


def resolve_rho(me, pd):
    """rho is a number, or {"from_calibration": <path under PROJ_DATA>} applying the frozen development rule.
    The calibration must have passed its checks."""
    r = me["rho"]
    if not isinstance(r, dict):
        return float(r), "config"
    path = os.path.join(pd, r["from_calibration"])
    cal = json.load(open(path))
    if cal.get("chosen_rho") is None or not cal.get("checks", {}).get("all_ok"):
        raise SystemExit(f"calibration {path} has no valid rho or failed its checks: {cal.get('checks')}")
    return float(cal["chosen_rho"]), path


def calibrate_radius(cfg, run, model, tok, params, theta0, h, disp, pairs, data, fmt, fit_seqs, fit_cls, sel,
                     ledger, res_dir):
    """Development-only radius rule: largest rho in the grid whose max over displacements of the median
    per-sequence sampled-token KL on kl_split prompts is <= kl_target."""
    me = cfg["mech"]
    probs = data[me["kl_split"]]
    texts = [chat_prompt(tok, task.prompt_text(p, cfg["prompt_variant"], fmt)) for p in probs]
    o = sample(model, tok, texts, 1, me["cap"], seed=me["sample_seed"] + 4242, batch_seqs=512)
    seqs = [(x["prompt_ids"], x["completion_ids"]) for x in o]
    fit0 = anchor_ll(model, fit_seqs, fit_cls, tok.pad_token_id, me["cap"], me["micro"])["mean_norm_ll"]
    grid = []
    for rho in me["rho_grid"]:
        row = {"rho": rho, "kl_median": {}, "taylor": {}}
        for (pi, arm), d in disp.items():
            kl = sampled_kl(model, params, theta0, rho * d, seqs, tok.pad_token_id, me["micro"])
            row["kl_median"][f"p{pi}-{arm}"] = float(np.median(kl))
            set_flat(params, theta0 + rho * d)
            fit1 = anchor_ll(model, fit_seqs, fit_cls, tok.pad_token_id, me["cap"], me["micro"])["mean_norm_ll"]
            set_flat(params, theta0)
            row["taylor"][f"p{pi}-{arm}"] = {"pred": float(torch.dot(h, rho * d)), "actual": fit1 - fit0}
        row["stat"] = max(row["kl_median"].values()) if row["kl_median"] else None
        grid.append(row)
        ledger.flush(note=f"calibrate rho {rho}")
        print(f"[calib] rho {rho}: stat {row['stat']}", flush=True)
    ok = [r["rho"] for r in grid if r["stat"] is not None and r["stat"] <= me["kl_target"]]
    chosen = max(ok) if ok else None
    checks = {"n_displacements": len(disp), "h_norm_positive": bool(float(h.norm()) > 0),
              "finite": bool(all(torch.isfinite(d).all() for d in disp.values()))}
    if chosen is not None:
        row = next(r for r in grid if r["rho"] == chosen)
        ratios = {k: (v["actual"] / v["pred"] if v["pred"] else None) for k, v in row["taylor"].items()}
        checks["taylor_ratio_at_chosen"] = ratios
        checks["taylor_ok"] = bool(all(x is not None and 0.5 <= x <= 2.0 for x in ratios.values()))
    checks["all_ok"] = bool(chosen is not None and checks["n_displacements"] >= 2 and checks["h_norm_positive"]
                            and checks["finite"] and checks.get("taylor_ok", False))
    out = {"run_id": run.id, "checks": checks, "rule": f"largest rho in {me['rho_grid']} with max-over-displacements median "
                                     f"sampled-token KL <= {me['kl_target']}", "chosen_rho": chosen, "grid": grid,
           "pairs": pairs, "selection": sel, "gpu_s_total": ledger.total_gpu_s()}
    assert torch.equal(flat_params(params), theta0)
    atomic_json(run.path("calibration.json"), out)
    if res_dir:
        atomic_json(os.path.join(res_dir, "calibration.json"), out)
    atomic_json(run.path("DONE.json"), {"config_hash": run.chash, "artifacts": []})
    ledger.flush(note="calibration complete")
    print(f"[calib] chosen rho = {chosen}")
    return 0


# ----------------------------------------------------------------------------- A5 / X6 validity checks

def fit_ab(rhos, actual):
    """Least-squares actual = a*rho + b*rho^2 (no intercept)."""
    x = np.asarray(rhos, float)
    a, b = np.linalg.lstsq(np.stack([x, x ** 2], 1), np.asarray(actual, float), rcond=None)[0]
    return float(a), float(b)


def validity_at(rho, rows, pairs, noise, rand_rows=None, kl_stat=None, kl_target=0.01, n_disp=None):
    """Checks V1-V5 (protocol A5 Section 4.2) at one radius.
    rows: {disp: [{"rho", "pred", "actual"}...]} over the grid (anchor-fit); pairs: list of (H key, L key);
    noise: fp32 repeat |difference|; rand_rows: same structure for random directions; kl_stat: max median KL at rho."""
    at = {k: next(r for r in v if r["rho"] == rho) for k, v in rows.items()}
    preds = [abs(r["pred"]) for r in at.values()]
    out = {"rho": rho, "noise": noise}
    out["V1"] = bool(noise <= 0.05 * float(np.median(preds)))
    v2a = []
    for hk, lk in pairs:
        pc = at[hk]["pred"] - at[lk]["pred"]
        ac = at[hk]["actual"] - at[lk]["actual"]
        v2a.append({"pair": f"{hk}/{lk}", "pred_contrast": pc, "actual_contrast": ac,
                    "abs_err_over_pred": abs(ac - pc) / abs(pc) if pc else None})
    rel = [abs(r["actual"] - r["pred"]) / abs(r["pred"]) if r["pred"] else float("inf") for r in at.values()]
    out["V2_pairs"] = v2a
    out["V2_median_rel_err"] = float(np.median(rel))
    out["V2"] = bool(all(x["abs_err_over_pred"] is not None and x["abs_err_over_pred"] <= 0.25 for x in v2a)
                     and out["V2_median_rel_err"] <= 0.25)
    curv = {}
    for k, v in rows.items():
        a, b = fit_ab([r["rho"] for r in v], [r["actual"] for r in v])
        curv[k] = {"a": a, "b": b, "quad_over_lin": abs(b) * rho ** 2 / max(abs(a) * rho, 1e-30)}
    out["V3_curvature"] = curv
    n_ok = sum(c["quad_over_lin"] <= 0.2 for c in curv.values())
    out["V3"] = bool(n_ok >= len(curv) - 1)
    if rand_rows is not None:
        rr = {k: next(r for r in v if r["rho"] == rho) for k, v in rand_rows.items()}
        out["V4_ratios"] = {k: (r["actual"] / r["pred"] if r["pred"] else None) for k, r in rr.items()}
        out["V4"] = bool(all(x is not None and 0.8 <= x <= 1.25 for x in out["V4_ratios"].values()))
    if kl_stat is not None:
        out["V5_kl_stat"] = kl_stat
        out["V5"] = bool(kl_stat <= kl_target)
    keys = [k for k in ("V1", "V2", "V3", "V4", "V5") if k in out]
    out["all_ok"] = bool(all(out[k] for k in keys))
    return out


def ll_grid(meas, mparams, theta0, dirs, seqs, cls, pad_id, Lmax, micro, grid, hvec):
    f0 = anchor_ll(meas, seqs, cls, pad_id, Lmax, micro)["mean_norm_ll"]
    rows = {}
    for k, d in dirs.items():
        rows[k] = []
        for rho in grid:
            set_flat(mparams, theta0 + rho * d)
            f1 = anchor_ll(meas, seqs, cls, pad_id, Lmax, micro)["mean_norm_ll"]
            set_flat(mparams, theta0)
            rows[k].append({"rho": rho, "pred": float(torch.dot(hvec, rho * d)), "actual": f1 - f0})
    return f0, rows


def checkpoint_validity(me, meas, mparams, theta0, h, disp, fit_seqs, fit_cls, pad_id, rho):
    """Study-checkpoint re-check of V1-V3 on anchor-fit only, at the calibrated rho, over the calibration grid."""
    grid = sorted(set(me["rho_grid"]) | {rho})
    a = anchor_ll(meas, fit_seqs, fit_cls, pad_id, me["cap"], me["micro"])["mean_norm_ll"]
    b = anchor_ll(meas, fit_seqs, fit_cls, pad_id, me["cap"], me["micro"])["mean_norm_ll"]
    dirs = {f"p{k[0]}-{k[1]}": d for k, d in disp.items()}
    _, rows = ll_grid(meas, mparams, theta0, dirs, fit_seqs, fit_cls, pad_id, me["cap"], me["micro"], grid, h)
    pairs = sorted({(f"p{k[0]}-H", f"p{k[0]}-L") for k in disp})
    checks = validity_at(rho, rows, pairs, abs(a - b))
    assert torch.equal(flat_params(mparams), theta0)
    return {"checks": checks, "grid": grid, "rows": rows}


def calibrate_grid(cfg, run, model, meas, tok, params, mparams, theta0, h, disp, pairs, data, fmt, fit_seqs, fit_cls,
                   sel, ledger, res_dir, enc):
    """Development-only X6 calibration: V1-V5 at every rho of the grid (anchor-fit gates), plus the held-out
    reference-trace diagnostic (no gate). The cross-checkpoint choice is made by scripts/x6_gate.py."""
    me = cfg["mech"]
    pad, cap, micro = tok.pad_token_id, me["cap"], me["micro"]
    grid = sorted(me["rho_grid"])
    held_seqs, held_cls = enc(data[me["heldout_split"]])
    h_held = anchor_direction(meas, mparams, held_seqs, pad, cap, micro)
    rep = {}
    for nm, (sq, cl) in (("fit", (fit_seqs, fit_cls)), ("held", (held_seqs, held_cls))):
        x = anchor_ll(meas, sq, cl, pad, cap, micro)["mean_norm_ll"]
        y = anchor_ll(meas, sq, cl, pad, cap, micro)["mean_norm_ll"]
        rep[nm] = abs(x - y)
    noise = max(rep.values())
    dirs = {f"p{k[0]}-{k[1]}": d for k, d in disp.items()}
    norms = [float(d.norm()) for d in disp.values()]
    g = torch.Generator().manual_seed(me.get("random_seed", 12345))
    rand = {}
    for i in range(me.get("n_random", 4)):
        r = torch.sign(torch.randn(theta0.numel(), generator=g)).to(theta0.device)
        rand[f"rand{i}"] = r * (float(np.median(norms)) / float(r.norm()))
    _, fit_rows = ll_grid(meas, mparams, theta0, dirs, fit_seqs, fit_cls, pad, cap, micro, grid, h)
    _, rand_rows = ll_grid(meas, mparams, theta0, rand, fit_seqs, fit_cls, pad, cap, micro, grid, h)
    _, held_rows = ll_grid(meas, mparams, theta0, dirs, held_seqs, held_cls, pad, cap, micro, grid, h_held)
    ledger.flush(note="calibrate2 likelihood grid")
    probs = data[me["kl_split"]]
    texts = [chat_prompt(tok, task.prompt_text(p, cfg["prompt_variant"], fmt)) for p in probs]
    o = sample(model, tok, texts, 1, cap, seed=me["sample_seed"] + 4242, batch_seqs=512)
    kseqs = [(x["prompt_ids"], x["completion_ids"]) for x in o]
    kl = {}
    for rho in grid:
        kl[rho] = max(float(np.median(sampled_kl(model, params, theta0, rho * d, kseqs, pad, micro)))
                      for d in disp.values())
    ledger.flush(note="calibrate2 KL")
    pk = sorted({(f"p{k[0]}-H", f"p{k[0]}-L") for k in disp})
    per_rho = {str(rho): validity_at(rho, fit_rows, pk, noise, rand_rows, kl[rho], me["kl_target"]) for rho in grid}
    diag = {}
    for rho in grid:
        at = {k: next(r for r in v if r["rho"] == rho) for k, v in held_rows.items()}
        signs = [(at[hk]["actual"] - at[lk]["actual"]) * (at[hk]["pred"] - at[lk]["pred"]) > 0 for hk, lk in pk]
        diag[str(rho)] = {"heldout_contrast_sign_agree": int(sum(signs)), "n_pairs": len(pk),
                          "heldout_ratio": {k: (r["actual"] / r["pred"] if r["pred"] else None) for k, r in at.items()}}
    out = {"run_id": run.id, "adapter": me["adapter"], "grid": grid, "noise_repeat": rep, "per_rho": per_rho,
           "passing_rhos": [rho for rho in grid if per_rho[str(rho)]["all_ok"]], "heldout_diagnostic": diag,
           "fit_rows": fit_rows, "rand_rows": rand_rows, "held_rows": held_rows, "pairs": pairs, "selection": sel,
           "disp_norms": norms, "gpu_s_total": ledger.total_gpu_s()}
    assert torch.equal(flat_params(params), theta0) and torch.equal(flat_params(mparams), theta0)
    atomic_json(run.path("calibration.json"), out)
    if res_dir:
        atomic_json(os.path.join(res_dir, "calibration.json"), out)
    atomic_json(run.path("DONE.json"), {"config_hash": run.chash, "artifacts": []})
    ledger.flush(note="calibrate2 complete")
    print(f"[calib2] passing rho: {out['passing_rhos']}; "
          f"{ {r: {k: v[k] for k in ('V1', 'V2', 'V3', 'V4', 'V5')} for r, v in per_rho.items()} }", flush=True)
    return 0
