"""`bc2rl online`: GRPO or online filtered supervision from a fixed initialisation.

One rollout round = P prompts x G samples (128 completions). At most one optimizer step per round.
Rounds are counted even when the step is skipped (no mixed group / no success).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

from . import task
from .data import load_data
from .evaluate import run_attempts, stable_int, summarize
from .losses import filtered_loss_sum, grpo_advantages, grpo_loss_sum
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
        old_lp, entropy = completion_logprob_table(model, seqs, tok.pad_token_id, cap, micro=on["micro"])
        adv = grpo_advantages(rewards, G) if on["method"] == "grpo" else None
        if on["method"] == "grpo":
            use = (adv != 0).nonzero().squeeze(-1).tolist()
        elif on["method"] == "filtered":
            use = (rewards > 0).nonzero().squeeze(-1).tolist()
        else:
            raise SystemExit(f"unknown method {on['method']}")
        skipped = len(use) == 0
        gn, ratio_dev, loss_total = None, 0.0, 0.0
        tp_max, tp_sum, tp_n, tp_out = 0.0, 0.0, 0.0, 0.0     # two-pass (no-grad vs grad) log-prob diagnostic
        if not skipped:
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
        else:
            counters["skipped"] += 1
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
                     "gen_tokens": gen_tok, "opt_steps": counters["opt_steps"],
                     "by_inputs": {str(h): float(np.mean([v["reward"] for v, p in
                                                          zip(ver, [probs[o["prompt_index"]] for o in outs])
                                                          if p.n_inputs == h] or [float("nan")]))
                                   for h in (3, 5, 10)},
                     "round_s": time.time() - t0, "gen_s": t_gen})
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
    save_state("complete")
    summary = {"run_id": run.id, "config_hash": run.chash, "method": on["method"], "init": init,
               "init_sha256": init_sha, "rounds": rnd, "counters": counters, "checkpoints": ckpts,
               "gpu_s_total": ledger.total_gpu_s(), "segments": ledger.data["segments"],
               "lr": on["lr"], "data_version": cfg["data"]["version"]}
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
    return 0
