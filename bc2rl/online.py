"""`bc2rl online`: GRPO or online filtered supervision from a fixed initialisation.

One rollout round = P prompts x G samples (128 completions). At most one optimizer step per round.
Rounds are counted even when the step is skipped (no mixed group / no success).

A6 options (protocol/AMENDMENT-A6 Section 6; absent = frozen behaviour):
  online.pool_inputs  = [3]  -> the prompt pool keeps only problems with these input counts (arm T3);
  online.train_inputs = [3]  -> only completions of these input counts carry gradient (arm MASK); every prompt
                                is still sampled, and the loss normalization is unchanged.

A6 Section 12 (GSPO stage; absent = frozen behaviour):
  online.objective = grpo_const (the GRPO above, K = 1 only) | grpo_len | gspo | ctpo (`losses.py`); setting it also turns
                     on the per-round diagnostics (update norm, steps.csv).
  online.updates_per_batch = K -> K full-batch optimizer steps per rollout batch (same micro-batch layout each step).
                     Behaviour log-probs are the detached log-probs of the first gradient pass of the round, so the
                     ratio of step 1 is exactly 1; they stay fixed for steps 2..K.
  online.clip_low / clip_high -> asymmetric clip band (default `clip`; null = no clipping, calibration probes only).
                     For ctpo they are log-space base values; online.clip_c (dual clip, default clip_high) and
                     online.clip_power (position exponent p, default 0.5; 0 = fixed band) complete the CTPO-LA band.
  online.train_precision = fp32 -> the frozen base weights are cast to FP32 (exact) for every scoring and training
                     pass and back to BF16 for sampling; TF32 is off. Default bf16.
  online.kl_every = N -> every N rounds, KL(current || initial adapter) on the rollout batch (same pass precision).
  online.dump_ratios = true -> per round, the signed token and sequence log-ratios of rows with A != 0 at each step
                     (`ratios/roundNNNN.npz` in the run directory; clip calibration probes, Section 12.4).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch

from . import task
from .data import load_data
from .evaluate import run_attempts, stable_int, summarize
from .losses import (CTPO_BUCKETS, ctpo_loss_sum, filtered_loss_sum, grpo_advantages, grpo_len_loss_sum,
                     grpo_loss_sum, gspo_loss_sum)
from .modeling import (chat_prompt, completion_logprob_table, gather_completion, load_adapter, load_policy,
                       load_tokenizer, model_info, sample, save_adapter, seq_logprobs, trainable_summary)
from .runtime import (CKPT_INTERVAL_S, CostLedger, Run, StopFlag, atomic_json, atomic_torch_save, rng_state,
                      set_rng_state, write_jsonl_gz)


def prompt_stream(n_pool, per_round, rounds, seed):
    """Deterministic prompt indices per round from the seed only, so both objectives and every
    initialisation with the same seed see the same prompt stream."""
    rng = np.random.default_rng(seed)
    need = per_round * rounds
    idx = []
    while len(idx) < need:
        idx.extend(rng.permutation(n_pool).tolist())
    return [idx[r * per_round:(r + 1) * per_round] for r in range(rounds)]


def set_frozen_dtype(model, dtype):
    """Cast the frozen (base) weights only; trainable adapter weights stay FP32. BF16 -> FP32 -> BF16 is exact."""
    for p in model.parameters():
        if not p.requires_grad and p.dtype != dtype:
            p.data = p.data.to(dtype)


def _q(v, qs=(0.5, 0.9, 0.99)):
    if v.numel() == 0:
        return [float("nan")] * (len(qs) + 1)
    v = v.float().cpu()
    return [float(torch.quantile(v, q)) if v.numel() < 16_000_000 else float("nan") for q in qs] + [float(v.max())]


def cmd_online(cfg):
    run = Run(cfg)
    if run.done_ok():
        print(f"[online] {run.id} already complete")
        return 0
    stop = StopFlag()
    on, val = cfg["online"], cfg["val"]
    pd = cfg["proj_data"]
    device = cfg.get("device", "cuda")
    data, man = load_data(cfg, [on["prompt_split"], val["split"]])
    pool, vprobs, fmt = data[on["prompt_split"]], data[val["split"]], data["format"][0]
    if on.get("pool_inputs"):
        pool = [p for p in pool if p.n_inputs in on["pool_inputs"]]
    train_inputs = set(on["train_inputs"]) if on.get("train_inputs") else None
    info = model_info(pd)
    tok = load_tokenizer(info["path"])
    model = load_policy(cfg.get("model_path_override") or info["path"], cfg["model"]["lora"], seed=cfg["seed"], device=device)
    init = on["init"]
    init_sha = "fresh-lora-B0"
    if init["adapter"] != "none":
        ipath = init["adapter"] if os.path.isabs(init["adapter"]) else os.path.join(pd, init["adapter"])
        init_sha = load_adapter(model, ipath)
        if init.get("sha256") and init["sha256"] != init_sha:
            raise SystemExit(f"init adapter hash {init_sha} != expected {init['sha256']}")
    init_state = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    objective = on.get("objective", "grpo_const") if on["method"] == "grpo" else None
    diag = "objective" in on
    K = int(on.get("updates_per_batch", 1))
    if objective not in (None, "grpo_const", "grpo_len", "gspo", "ctpo"):
        raise SystemExit(f"unknown objective {objective}")
    if K != 1 and objective in (None, "grpo_const"):
        raise SystemExit("updates_per_batch > 1 needs objective grpo_len, gspo or ctpo")
    if objective in ("grpo_len", "gspo", "ctpo") and not ("clip_low" in on and "clip_high" in on):
        raise SystemExit(f"objective {objective} needs explicit clip_low and clip_high (null = no clipping)")
    _b = lambda k: math.inf if on.get(k, on.get("clip")) is None else float(on.get(k, on.get("clip")))
    e_lo, e_hi = _b("clip_low"), _b("clip_high")
    e_c = (math.inf if on["clip_c"] is None else float(on["clip_c"])) if "clip_c" in on else e_hi
    c_pow = float(on.get("clip_power", 0.5))
    clip_diag = float(on.get("clip") or 0.2)          # band of the two-pass diagnostic only
    train_fp32 = on.get("train_precision", "bf16") == "fp32"
    if on.get("train_precision", "bf16") not in ("bf16", "fp32"):
        raise SystemExit(f"unknown train_precision {on['train_precision']}")
    if train_fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    kl_every = on.get("kl_every")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=on["lr"], weight_decay=0.0)     # fresh optimizer state at the fork
    P, G, R, cap = on["prompts_per_round"], on["group"], on["rounds"], on["cap"]
    Q = P * G
    stream = prompt_stream(len(pool), P, R, on.get("stream_seed", cfg["seed"]))
    texts = [chat_prompt(tok, task.prompt_text(p, cfg["prompt_variant"], fmt)) for p in pool]
    ledger = CostLedger(run.path("cost.json"), "online")
    state_path = run.path("state", "state.pt")
    rnd, rows, ckpts, counters = 0, [], [], {"opt_steps": 0, "generated_tokens": 0, "processed_tokens": 0,
                                             "skipped": 0}
    if diag:
        counters["trained_tokens"] = 0          # tokens through a gradient pass, summed over the K steps
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        load_adapter(model, st["adapter_path"])
        opt.load_state_dict(st["opt"])
        set_rng_state(st["rng"])
        rnd, rows, ckpts, counters = st["round"], st["rows"], st["ckpts"], st["counters"]
        print(f"[online] resumed {run.id} at round {rnd}", flush=True)
    else:
        atomic_json(run.path("init.json"), {"init": init, "init_sha256": init_sha})
    atomic_json(run.path("trainable.json"), trainable_summary(model))
    tag_seed = cfg["seed"] * 7919 + stable_int(f"{on['method']}:{init.get('tag', init['adapter'])}")

    def save_state(note):
        apath = run.path("state", "adapter.safetensors")
        save_adapter(model, apath)
        atomic_torch_save(state_path, {"round": rnd, "opt": opt.state_dict(), "rng": rng_state(), "rows": rows,
                                       "ckpts": ckpts, "counters": counters, "adapter_path": apath,
                                       "config_hash": run.chash, "scheduler": "constant"})
        with open(run.path("metrics.jsonl.tmp"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(run.path("metrics.jsonl.tmp"), run.path("metrics.jsonl"))
        ledger.flush(note=note)

    def validate(r_now):
        model.eval()
        vrec, gen_s = run_attempts(model, tok, vprobs, fmt, cfg["prompt_variant"], val["attempts"], val["cap"],
                                   seed=tag_seed + 100000 + r_now)
        write_jsonl_gz(run.path("val", f"round{r_now:04d}.jsonl.gz"), vrec)
        return summarize(vrec), gen_s

    if rnd == 0 and 0 in on["val_rounds"] and not any(c["round"] == 0 for c in ckpts):
        vs, gen_s = validate(0)
        ckpts.append({"round": 0, "adapter": None, "sha256": init_sha, "val": vs, "val_gen_s": gen_s})
        print(f"[online] round 0 val {json.dumps(vs['metrics'])}", flush=True)
        save_state("round 0 validation")
    last_ckpt = time.time()
    while rnd < R:
        t0 = time.time()
        probs = [pool[i] for i in stream[rnd]]
        model.eval()
        outs = sample(model, tok, [texts[i] for i in stream[rnd]], G, cap, seed=tag_seed + rnd, batch_seqs=Q)
        t_gen = time.time() - t0
        ver = [task.verify(probs[o["prompt_index"]].inputs, probs[o["prompt_index"]].target, o["text"],
                           truncated=(o["finish"] == "length")) for o in outs]
        rewards = torch.tensor([v["reward"] for v in ver], dtype=torch.float32)
        if cfg.get("test_mode") and on.get("debug_reward") == "random":
            g = torch.Generator().manual_seed(tag_seed + rnd)
            rewards = (torch.rand(Q, generator=g) < 0.5).float()
        grp = rewards.view(P, G).sum(1)
        n_mixed = int(((grp > 0) & (grp < G)).sum())
        n_all = int((grp == G).sum())
        n_none = int((grp == 0).sum())
        seqs = [(o["prompt_ids"], o["completion_ids"]) for o in outs]
        ntok = torch.tensor([o["n_tokens"] for o in outs], device=model.device)
        tmask = (torch.arange(cap, device=model.device).unsqueeze(0) < ntok.unsqueeze(1)).float()
        if train_fp32:
            set_frozen_dtype(model, torch.float32)
        old_lp, entropy = completion_logprob_table(model, seqs, tok.pad_token_id, cap, micro=on["micro"])
        kl = None
        if kl_every and rnd % kl_every == 0:
            with torch.no_grad():
                cur = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        p.copy_(init_state[n])
                init_lp, _ = completion_logprob_table(model, seqs, tok.pad_token_id, cap, micro=on["micro"])
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        p.copy_(cur[n])
                d = (old_lp - init_lp) * tmask
                nt = float(tmask.sum())
                kl = {"kl_after_updates_of_rounds": rnd, "kl_k1_tok": float(d.sum()) / max(nt, 1.0),
                      "kl_k3_tok": float(((torch.exp(-d) - 1 + d) * tmask).sum()) / max(nt, 1.0),
                      "kl_seq": float(d.sum()) / Q}
        adv = grpo_advantages(rewards, G) if on["method"] == "grpo" else None
        if on["method"] == "grpo":
            use = (adv != 0).nonzero().squeeze(-1).tolist()
        elif on["method"] == "filtered":
            use = (rewards > 0).nonzero().squeeze(-1).tolist()
        else:
            raise SystemExit(f"unknown method {on['method']}")
        if train_inputs is not None:
            use = [k for k in use if probs[outs[k]["prompt_index"]].n_inputs in train_inputs]
        used_by_inputs = {str(h): sum(probs[outs[k]["prompt_index"]].n_inputs == h for k in use) for h in (3, 5, 10)}
        skipped = len(use) == 0
        gn, ratio_dev, loss_total, steps, upd = None, 0.0, 0.0, [], None
        tp_max, tp_sum, tp_n, tp_out = 0.0, 0.0, 0.0, 0.0     # two-pass (no-grad vs grad) log-prob diagnostic
        if not skipped and objective in ("grpo_len", "gspo", "ctpo"):
            model.train()
            sub = [seqs[i] for i in use]
            beh, dump = {}, {}
            for k in range(K):
                opt.zero_grad(set_to_none=True)
                p0 = [p.detach().clone() for p in params]
                st = {"units": 0.0, "outside": 0.0, "clipped": 0.0, "clipped_tokens": 0.0, "unit_tokens": 0.0}
                st.update({f"{x}{b0}": 0.0 for b0, _ in CTPO_BUCKETS for x in "uoc"})
                tok_lr, seq_lr, cum_lr, cum_pos, lsum = [], [], [], [], 0.0
                for j, (sl, lp, cm, _) in enumerate(seq_logprobs(model, sub, tok.pad_token_id, on["micro"], grad=True)):
                    rows_i = use[sl.start:sl.stop]
                    lp_g = gather_completion(lp, cm, cap)
                    m = tmask[rows_i]
                    if k == 0:
                        beh[j] = lp_g.detach()
                        dev = (lp_g.detach() - old_lp[rows_i]).abs() * m
                        tp_max = max(tp_max, float(dev.max()))
                        tp_sum += float(dev.sum())
                        tp_n += float(m.sum())
                        tp_out += float((((dev.exp() > 1 + clip_diag) | ((-dev).exp() < 1 - clip_diag)).float() * m).sum())
                    a_i = adv[rows_i].to(lp_g.device)
                    if objective == "ctpo":
                        ls, _, sts = ctpo_loss_sum(lp_g, beh[j], a_i, m, e_lo, e_hi, e_c, c_pow)
                    else:
                        fn = gspo_loss_sum if objective == "gspo" else grpo_len_loss_sum
                        ls, _, sts = fn(lp_g, beh[j], a_i, m, e_lo, e_hi)
                    loss = ls / Q
                    if not torch.isfinite(loss):
                        raise SystemExit(f"non-finite loss at round {rnd + 1} step {k + 1}")
                    loss.backward()
                    lsum += float(loss)
                    for key in st:
                        st[key] += sts.get(key, 0.0)
                    d = lp_g.detach() - beh[j]
                    nz = a_i != 0
                    tok_lr.append(d[(m > 0) & nz.unsqueeze(1)])
                    seq_lr.append(((d * m).sum(1) / m.sum(1).clamp(min=1.0))[nz & (m.sum(1) > 0)])
                    sel = (m > 0) & nz.unsqueeze(1)
                    cum_lr.append(torch.cumsum(d * m, dim=1)[sel])
                    cum_pos.append(torch.cumsum(m, dim=1)[sel])
                g_k = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                if not math.isfinite(g_k):
                    raise SystemExit(f"non-finite gradient norm at round {rnd + 1} step {k + 1}")
                opt.step()
                counters["opt_steps"] += 1
                counters["trained_tokens"] += int(sum(len(p) + len(c) for p, c in sub))
                u_k = float(sum(((p.detach() - q) ** 2).sum() for p, q in zip(params, p0))) ** 0.5
                tl, sq = torch.cat(tok_lr).abs(), torch.cat(seq_lr)
                cl, cp = torch.cat(cum_lr), torch.cat(cum_pos)
                if on.get("dump_ratios"):
                    dump[f"tok{k + 1}"] = torch.cat(tok_lr).float().cpu().numpy()
                    dump[f"seq{k + 1}"] = sq.float().cpu().numpy()
                    dump[f"cum{k + 1}"] = cl.float().cpu().numpy()
                    dump[f"pos{k + 1}"] = cp.to(torch.int16).cpu().numpy()
                by_pos = {}
                for b0, b1 in CTPO_BUCKETS:
                    v = cl[(cp >= b0) & (cp <= b1)].abs()
                    by_pos[str(b0)] = {"n": int(v.numel()), "cum_abs": _q(v, (0.5, 0.9))}
                    if objective == "ctpo":
                        by_pos[str(b0)].update(frac_outside=st[f"o{b0}"] / max(st[f"u{b0}"], 1.0),
                                               frac_clipped=st[f"c{b0}"] / max(st[f"u{b0}"], 1.0))
                steps.append({"step": k + 1, "loss": lsum, "grad_norm": g_k, "update_norm": u_k,
                              "tok_abslogr": _q(tl), "seq_abslogs": _q(sq.abs()), "cum_abslogr": _q(cl.abs()),
                              "seq_logs_mean": float(sq.mean()) if sq.numel() else float("nan"),
                              "units": st["units"], "frac_outside": st["outside"] / max(st["units"], 1.0),
                              "frac_clipped": st["clipped"] / max(st["units"], 1.0),
                              "frac_tokens_clipped": (st["clipped_tokens"] / max(st["unit_tokens"], 1.0)
                                                      if objective == "gspo" else st["clipped"] / max(st["units"], 1.0)),
                              "by_pos": by_pos})
                ratio_dev = max(ratio_dev, float(torch.expm1(tl.max()).abs()) if tl.numel() else 0.0)
            gn, loss_total = steps[0]["grad_norm"], steps[0]["loss"]
            if dump:
                os.makedirs(run.path("ratios"), exist_ok=True)
                np.savez_compressed(run.path("ratios", f"round{rnd + 1:04d}.npz"), **dump)
            upd = float(sum(((p.detach() - q) ** 2).sum() for p, q in zip(params, [init_state[n] for n, pp in
                        model.named_parameters() if pp.requires_grad]))) ** 0.5
        elif not skipped:
            p0 = [p.detach().clone() for p in params] if diag else None
            model.train()
            opt.zero_grad(set_to_none=True)
            sub = [seqs[i] for i in use]
            for sl, lp, cm, _ in seq_logprobs(model, sub, tok.pad_token_id, on["micro"], grad=True):
                rows_i = use[sl.start:sl.stop]
                lp_g = gather_completion(lp, cm, cap)
                m = tmask[rows_i]
                if on["method"] == "grpo":
                    # Old log-probs come from this same forward pass (detached): with one update per fresh
                    # batch the rollout policy equals the current parameters, so the ratio is exactly 1.
                    # A separate no-grad pass differs only by BF16 numerics (padding/batch shape); using it
                    # would clip tokens spuriously. Its deviation is logged as a stale-policy diagnostic.
                    ls, ratio = grpo_loss_sum(lp_g, lp_g.detach(), adv[rows_i].to(lp_g.device), m, on["clip"])
                    dev = (lp_g.detach() - old_lp[rows_i]).abs() * m
                    tp_max = max(tp_max, float(dev.max()))
                    tp_sum += float(dev.sum())
                    tp_n += float(m.sum())
                    tp_out += float((((dev.exp() > 1 + on["clip"]) | ((-dev).exp() < 1 - on["clip"])).float() * m).sum())
                    loss = ls / (Q * cap)
                    ratio_dev = max(ratio_dev, float(((ratio - 1).abs() * m).max()))
                else:
                    loss = filtered_loss_sum(lp_g, m) / (len(use) * cap)
                loss.backward()
                loss_total += float(loss)
            gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
            counters["opt_steps"] += 1
            if diag:
                counters["trained_tokens"] += int(sum(len(seqs[i][0]) + len(seqs[i][1]) for i in use))
                steps.append({"step": 1, "loss": loss_total, "grad_norm": gn, "update_norm": float(sum(
                    ((p.detach() - q) ** 2).sum() for p, q in zip(params, p0))) ** 0.5})
                upd = float(sum(((p.detach() - q) ** 2).sum() for p, q in zip(params, [init_state[n] for n, pp in
                            model.named_parameters() if pp.requires_grad]))) ** 0.5
        else:
            counters["skipped"] += 1
        if train_fp32:
            set_frozen_dtype(model, torch.bfloat16)
        gen_tok = int(ntok.sum())
        counters["generated_tokens"] += gen_tok
        counters["processed_tokens"] += int(sum(len(p) + len(c) for p, c in seqs)) + \
            (int(sum(len(seqs[i][0]) + len(seqs[i][1]) for i in use)) if not skipped else 0)
        rnd += 1
        errs = {}
        for v in ver:
            errs[v["error"]] = errs.get(v["error"], 0) + 1
        rows.append({"round": rnd, "reward_mean": float(rewards.mean()), "successes": int(rewards.sum()),
                     "mixed_groups": n_mixed, "all_success_groups": n_all, "all_fail_groups": n_none,
                     "skipped": skipped, "loss": loss_total, "grad_norm": gn, "ratio_max_dev": ratio_dev,
                     "twopass_absdev_max": tp_max, "twopass_absdev_mean": tp_sum / max(tp_n, 1.0),
                     "twopass_frac_outside_clip": tp_out / max(tp_n, 1.0),
                     "entropy": entropy, "mean_len": float(ntok.float().mean()),
                     "truncated": int(sum(o["finish"] == "length" for o in outs)),
                     "format_valid": float(np.mean([v["format_valid"] for v in ver])),
                     "valid_prefix": float(np.mean([v["valid_prefix"] for v in ver])), "errors": errs,
                     "gen_tokens": gen_tok, "opt_steps": counters["opt_steps"], "used_by_inputs": used_by_inputs,
                     "by_inputs": {str(h): float(np.mean([v["reward"] for v, p in
                                                          zip(ver, [probs[o["prompt_index"]] for o in outs])
                                                          if p.n_inputs == h] or [float("nan")]))
                                   for h in (3, 5, 10)},
                     "round_s": time.time() - t0, "gen_s": t_gen})
        if diag:
            rows[-1].update({"steps": steps, "dist_from_init": upd, **(kl or {})})
        write_jsonl_gz(run.path("records", f"round{rnd:04d}.jsonl.gz"),
                       [{"round": rnd, "pid": probs[o["prompt_index"]].pid, "attempt": o["attempt"],
                         "text": o["text"], "n_tokens": o["n_tokens"], "finish": o["finish"],
                         "reward": float(rewards[k]), "adv": (float(adv[k]) if adv is not None else None),
                         "error": ver[k]["error"], "seed": tag_seed + rnd} for k, o in enumerate(outs)])
        if rnd in on["save_rounds"] or rnd in on["val_rounds"]:
            entry = {"round": rnd, "adapter": None, "sha256": None}
            if rnd in on["save_rounds"]:
                ap = run.path("adapters", f"round{rnd:04d}.safetensors")
                entry["sha256"] = save_adapter(model, ap)
                entry["adapter"] = os.path.relpath(ap, pd)
            if rnd in on["val_rounds"]:
                entry["val"], entry["val_gen_s"] = validate(rnd)
                print(f"[online] round {rnd} val {json.dumps(entry['val']['metrics'])}", flush=True)
            entry["counters"] = dict(counters)
            entry["gpu_s_so_far"] = ledger.total_gpu_s()
            ckpts = [c for c in ckpts if c["round"] != rnd] + [entry]
            save_state(f"round {rnd}")
            last_ckpt = time.time()
        elif time.time() - last_ckpt > CKPT_INTERVAL_S or (cfg.get("test_mode") and
                                                            rnd % cfg.get("_debug_ckpt_every", 10 ** 9) == 0):
            save_state(f"round {rnd}")
            last_ckpt = time.time()
        if rnd % 8 == 0:
            r = rows[-1]
            print(f"[online] r{rnd} succ={r['successes']} mixed={r['mixed_groups']} skip={counters['skipped']} "
                  f"len={r['mean_len']:.0f} ent={r['entropy']:.3f} gn={r['grad_norm']} {r['round_s']:.1f}s", flush=True)
        if cfg.get("test_mode") and cfg.get("_debug_kill_after_round") == rnd:
            os._exit(17)
        if stop.stop:
            save_state(f"signal {stop.signum} at round {rnd}")
            print(f"[online] signal {stop.signum}: checkpointed at round {rnd}; exiting for requeue", flush=True)
            sys.exit(143)
    kl_final = None
    if kl_every:
        # KL of the final adapter (after all R rounds) on one extra rollout batch sampled from it; the per-round KL
        # entries measure the adapter before that round's update (after `kl_after_updates_of_rounds` rounds).
        model.eval()
        fin_idx = prompt_stream(len(pool), P, R + 1, on.get("stream_seed", cfg["seed"]))[R]
        fo = sample(model, tok, [texts[i] for i in fin_idx], G, cap, seed=tag_seed + 10 ** 6, batch_seqs=Q)
        fseqs = [(o["prompt_ids"], o["completion_ids"]) for o in fo]
        fn_ = torch.tensor([o["n_tokens"] for o in fo], device=model.device)
        fm = (torch.arange(cap, device=model.device).unsqueeze(0) < fn_.unsqueeze(1)).float()
        if train_fp32:
            set_frozen_dtype(model, torch.float32)
        with torch.no_grad():
            cur_lp, _ = completion_logprob_table(model, fseqs, tok.pad_token_id, cap, micro=on["micro"])
            cur = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(init_state[n])
            ini_lp, _ = completion_logprob_table(model, fseqs, tok.pad_token_id, cap, micro=on["micro"])
            for n, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(cur[n])
        if train_fp32:
            set_frozen_dtype(model, torch.bfloat16)
        d = (cur_lp - ini_lp) * fm
        nt = float(fm.sum())
        kl_final = {"kl_after_updates_of_rounds": R, "kl_k1_tok": float(d.sum()) / max(nt, 1.0),
                    "kl_k3_tok": float(((torch.exp(-d) - 1 + d) * fm).sum()) / max(nt, 1.0), "kl_seq": float(d.sum()) / Q,
                    "mean_len": float(fn_.float().mean())}
    save_state("complete")
    summary = {"run_id": run.id, "config_hash": run.chash, "method": on["method"], "init": init,
               "init_sha256": init_sha, "rounds": rnd, "counters": counters, "checkpoints": ckpts,
               "gpu_s_total": ledger.total_gpu_s(), "segments": ledger.data["segments"],
               "lr": on["lr"], "data_version": cfg["data"]["version"]}
    if kl_final is not None:
        summary["kl_final"] = kl_final
    atomic_json(run.path("summary.json"), summary)
    atomic_json(run.path("DONE.json"), {"config_hash": run.chash, "artifacts": [
        {"path": os.path.relpath(os.path.join(pd, c["adapter"]), run.root), "sha256": c["sha256"]}
        for c in ckpts if c.get("adapter")]})
    res = cfg.get("_results_dir")
    if res:
        atomic_json(os.path.join(res, "summary.json"), summary)
        keys = ["round", "reward_mean", "successes", "mixed_groups", "all_success_groups", "all_fail_groups",
                "skipped", "loss", "grad_norm", "ratio_max_dev", "entropy", "mean_len", "truncated", "format_valid",
                "valid_prefix", "gen_tokens", "opt_steps", "round_s", "gen_s", "twopass_absdev_max",
                "twopass_absdev_mean", "twopass_frac_outside_clip"]
        with open(os.path.join(res, "rounds.csv"), "w") as f:
            f.write(",".join(keys) + ",r3,r5,r10\n")
            for r in rows:
                f.write(",".join(str(r[k]) for k in keys) + "," +
                        ",".join(f"{r['by_inputs'][h]:.4f}" for h in ("3", "5", "10")) + "\n")
        if diag:
            f4 = lambda x: "" if x is None else (f"{x:.4g}" if isinstance(x, float) else str(x))
            sk = ["loss", "grad_norm", "update_norm", "seq_logs_mean", "units", "frac_outside", "frac_clipped",
                  "frac_tokens_clipped"]
            with open(os.path.join(res, "steps.csv"), "w") as f:
                f.write("round,step," + ",".join(sk) + ",tok_p50,tok_p90,tok_p99,tok_max,seq_p50,seq_p90,seq_p99,"
                        "seq_max,dist_from_init,kl_k1_tok,kl_k3_tok,kl_seq\n")
                for r in rows:
                    for s_ in r.get("steps") or ([{"step": 0}] if r.get("kl_seq") is not None else []):
                        f.write(f"{r['round']},{s_['step']}," + ",".join(f4(s_.get(k)) for k in sk) + "," +
                                ",".join(f4(x) for x in (s_.get("tok_abslogr") or [None] * 4) +
                                         (s_.get("seq_abslogs") or [None] * 4)) + "," +
                                ",".join(f4(r.get(k)) for k in ("dist_from_init", "kl_k1_tok", "kl_k3_tok",
                                                                 "kl_seq")) + "\n")
            # position buckets: per 32-round block, step and bucket, the mean of the per-round values
            agg = {}
            for r in rows:
                for s_ in r.get("steps") or []:
                    for b0, v in (s_.get("by_pos") or {}).items():
                        key = ((r["round"] - 1) // 32 + 1, s_["step"], int(b0))
                        a_ = agg.setdefault(key, {"n": 0, "rounds": 0, "cum_p50": 0.0, "cum_p90": 0.0,
                                                  "frac_outside": 0.0, "frac_clipped": 0.0})
                        if v["n"] == 0:
                            continue
                        a_["rounds"] += 1
                        a_["n"] += v["n"]
                        a_["cum_p50"] += v["cum_abs"][0]
                        a_["cum_p90"] += v["cum_abs"][1]
                        a_["frac_outside"] += v.get("frac_outside", float("nan"))
                        a_["frac_clipped"] += v.get("frac_clipped", float("nan"))
            if agg:
                with open(os.path.join(res, "pos.csv"), "w") as f:
                    f.write("block32,step,pos_from,rounds,tokens,cum_abs_p50,cum_abs_p90,frac_outside,frac_clipped\n")
                    for (blk, stp, b0), v in sorted(agg.items()):
                        n = max(v["rounds"], 1)
                        f.write(f"{blk},{stp},{b0},{v['rounds']},{v['n']}," + ",".join(
                            f"{v[k] / n:.4g}" for k in ("cum_p50", "cum_p90", "frac_outside", "frac_clipped")) + "\n")
    return 0
