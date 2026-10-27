"""A6 Section 12.3 GPU validation (V3, V4, V7) on the real model and the dev-seed initial adapters.

For each initial adapter (light, extended of the dev seed) and each of `--batches` real rollout batches (the online
stream of the frozen M-arm configuration, rounds 0, 1, ...):
  V7 determinism: two BF16 no-grad passes with the same layout give identical log-probs (max |diff|).
  V3 first update: the accumulated gradients of grpo_len, gspo and ctpo (reference bands) at behaviour = current
     (ratio 1), in FP32 and in BF16 (relative L2 difference and cosine against grpo_len).
  V4 precision: from FP32 behaviour log-probs, 3 full-batch GSPO steps (lr of the config, no clipping, FP32 passes).
     After each step the token and sequence log-ratios are computed with the same layout in FP32 and in BF16 (BF16
     behaviour log-probs from the BF16 pass before the first step). noise = BF16 log-ratio - FP32 log-ratio.
     Also the cumulative (prefix-sum) log-ratio used by CTPO.
     Decision (Section 12.3 / 12.8): BF16 if, for every adapter and batch, after step 1
       median |noise| <= 0.1 * median |FP32 log-ratio| at the sequence, token and cumulative level;
     otherwise FP32 for every arm of the stage.
  Timing: seconds per full-batch gradient pass in BF16 and FP32 (forecast input).
Writes g_validate.json to --out.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from bc2rl import task
from bc2rl.data import load_data
from bc2rl.evaluate import stable_int
from bc2rl.losses import ctpo_loss_sum, grpo_advantages, grpo_len_loss_sum, gspo_loss_sum
from bc2rl.modeling import (chat_prompt, gather_completion, load_adapter, load_policy, load_tokenizer, model_info,
                            sample, seq_logprobs)
from bc2rl.online import prompt_stream, set_frozen_dtype
from bc2rl.runtime import load_config


def logp_rows(model, sub, pad, micro, cap, grad=False):
    out = []
    for sl, lp, cm, _ in seq_logprobs(model, sub, pad, micro, grad=grad):
        out.append(gather_completion(lp, cm, cap))
    return out


def grad_vec(model, params, sub, pad, micro, cap, adv, m, Q, fn):
    for p in params:
        p.grad = None
    k = 0
    for sl, lp, cm, _ in seq_logprobs(model, sub, pad, micro, grad=True):
        lp_g = gather_completion(lp, cm, cap)
        rows = slice(sl.start, sl.stop)
        if fn is ctpo_loss_sum:
            ls, _, _ = fn(lp_g, lp_g.detach(), adv[rows], m[rows], 0.025, 0.05, 0.05, 0.5)
        else:
            ls, _, _ = fn(lp_g, lp_g.detach(), adv[rows], m[rows], float("inf"), float("inf"))
        (ls / Q).backward()
        k += 1
    return torch.cat([p.grad.detach().flatten().double() for p in params])


def read_config(path):
    """The project loader (JSON first; PyYAML reads 1e-05 as a string) with PROJ_DATA of the job."""
    cfg = load_config(path)
    cfg["proj_data"] = os.environ["PROJ_DATA"]
    return cfg


def med(x):
    return float(torch.median(x.abs().float().cpu())) if x.numel() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True, help="frozen online configs (one per initial adapter)")
    ap.add_argument("--batches", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    res = {"adapters": [], "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    for cpath in a.configs:
        cfg = read_config(cpath)
        on = cfg["online"]
        pd = cfg["proj_data"]
        data, _ = load_data(cfg, [on["prompt_split"]])
        pool, fmt = data[on["prompt_split"]], data["format"][0]
        info = model_info(pd)
        tok = load_tokenizer(info["path"])
        model = load_policy(info["path"], cfg["model"]["lora"], seed=cfg["seed"], device="cuda")
        init = on["init"]
        sha = load_adapter(model, os.path.join(pd, init["adapter"]))
        params = [p for p in model.parameters() if p.requires_grad]
        start = {id(p): p.detach().clone() for p in params}
        P, G, cap, micro = on["prompts_per_round"], on["group"], on["cap"], on["micro"]
        Q = P * G
        stream = prompt_stream(len(pool), P, a.batches, on.get("stream_seed", cfg["seed"]))
        texts = [chat_prompt(tok, task.prompt_text(p, cfg["prompt_variant"], fmt)) for p in pool]
        tag_seed = cfg["seed"] * 7919 + stable_int(f"{on['method']}:{init.get('tag', init['adapter'])}")
        entry = {"config": cpath, "init_sha256": sha, "batches": []}
        for b in range(a.batches):
            with torch.no_grad():
                for p in params:
                    p.copy_(start[id(p)])
            set_frozen_dtype(model, torch.bfloat16)
            probs = [pool[i] for i in stream[b]]
            outs = sample(model, tok, [texts[i] for i in stream[b]], G, cap, seed=tag_seed + b, batch_seqs=Q)
            ver = [task.verify(probs[o["prompt_index"]].inputs, probs[o["prompt_index"]].target, o["text"],
                               truncated=(o["finish"] == "length")) for o in outs]
            rewards = torch.tensor([v["reward"] for v in ver], dtype=torch.float32)
            adv_all = grpo_advantages(rewards, G)
            use = (adv_all != 0).nonzero().squeeze(-1).tolist()
            row = {"batch": b, "n_use": len(use), "mean_len": float(np.mean([o["n_tokens"] for o in outs]))}
            if not use:
                entry["batches"].append(row)
                continue
            sub = [(outs[i]["prompt_ids"], outs[i]["completion_ids"]) for i in use]
            ntok = torch.tensor([outs[i]["n_tokens"] for i in use], device=model.device)
            m = (torch.arange(cap, device=model.device).unsqueeze(0) < ntok.unsqueeze(1)).float()
            adv = adv_all[use].to(model.device)
            model.train()
            # V7 determinism (BF16)
            x1 = torch.cat(logp_rows(model, sub, tok.pad_token_id, micro, cap))
            x2 = torch.cat(logp_rows(model, sub, tok.pad_token_id, micro, cap))
            row["V7_bf16_repeat_maxabs"] = float(((x1 - x2) * m).abs().max())
            # V3 first-update gradients + timing
            for dt, name in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
                set_frozen_dtype(model, dt)
                torch.cuda.synchronize()
                t0 = time.time()
                gl = grad_vec(model, params, sub, tok.pad_token_id, micro, cap, adv, m, Q, grpo_len_loss_sum)
                torch.cuda.synchronize()
                row[f"pass_s_{name}"] = time.time() - t0
                for tag, fn in (("", gspo_loss_sum), ("ctpo_", ctpo_loss_sum)):
                    gs = grad_vec(model, params, sub, tok.pad_token_id, micro, cap, adv, m, Q, fn)
                    row[f"V3_{tag}{name}_rel_l2"] = float((gl - gs).norm() / gl.norm())
                    row[f"V3_{tag}{name}_cos"] = float(torch.dot(gl, gs) / (gl.norm() * gs.norm()))
            # V4 precision
            set_frozen_dtype(model, torch.bfloat16)
            b16 = [x.detach() for x in logp_rows(model, sub, tok.pad_token_id, micro, cap)]
            set_frozen_dtype(model, torch.float32)
            b32 = [x.detach() for x in logp_rows(model, sub, tok.pad_token_id, micro, cap)]
            opt = torch.optim.AdamW(params, lr=on["lr"], weight_decay=0.0)
            steps = []
            for k in range(a.steps):
                opt.zero_grad(set_to_none=True)
                j = 0
                for sl, lp, cm, _ in seq_logprobs(model, sub, tok.pad_token_id, micro, grad=True):
                    lp_g = gather_completion(lp, cm, cap)
                    rows = slice(sl.start, sl.stop)
                    ls, _, _ = gspo_loss_sum(lp_g, b32[j], adv[rows], m[rows], float("inf"), float("inf"))
                    (ls / Q).backward()
                    j += 1
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                r32 = torch.cat(logp_rows(model, sub, tok.pad_token_id, micro, cap)) - torch.cat(b32)
                set_frozen_dtype(model, torch.bfloat16)
                r16 = torch.cat(logp_rows(model, sub, tok.pad_token_id, micro, cap)) - torch.cat(b16)
                set_frozen_dtype(model, torch.float32)
                n = m.sum(1)
                s32, s16 = (r32 * m).sum(1) / n, (r16 * m).sum(1) / n
                tok32, tokn = r32[m > 0], (r16 - r32)[m > 0]
                c32, c16 = torch.cumsum(r32 * m, 1), torch.cumsum(r16 * m, 1)
                cum32, cumn = c32[m > 0], (c16 - c32)[m > 0]
                steps.append({"step": k + 1, "med_abs_logr_tok_fp32": med(tok32), "med_abs_noise_tok": med(tokn),
                              "med_abs_logs_seq_fp32": med(s32), "med_abs_noise_seq": med(s16 - s32),
                              "p90_abs_logs_seq_fp32": float(torch.quantile(s32.abs().float().cpu(), 0.9)),
                              "p90_abs_noise_seq": float(torch.quantile((s16 - s32).abs().float().cpu(), 0.9)),
                              "med_abs_cum_fp32": med(cum32), "med_abs_noise_cum": med(cumn)})
            row["V4_steps"] = steps
            s1 = steps[0]
            row["V4_bf16_ok"] = bool(s1["med_abs_noise_seq"] <= 0.1 * s1["med_abs_logs_seq_fp32"] and
                                     s1["med_abs_noise_tok"] <= 0.1 * s1["med_abs_logr_tok_fp32"] and
                                     s1["med_abs_noise_cum"] <= 0.1 * s1["med_abs_cum_fp32"])
            entry["batches"].append(row)
            print(f"[g-validate] {init.get('tag')} batch {b}: " + json.dumps({k: v for k, v in row.items()
                                                                             if k != "V4_steps"}), flush=True)
            print(f"[g-validate]   V4 steps: {json.dumps(steps)}", flush=True)
        res["adapters"].append(entry)
        del model
        torch.cuda.empty_cache()
    rows = [r for e in res["adapters"] for r in e["batches"] if "V4_bf16_ok" in r]
    res["V3_pass"] = bool(rows) and all(r[f"V3_{t}fp32_rel_l2"] <= 1e-4 and r[f"V3_{t}fp32_cos"] >= 0.99999
                                        for r in rows for t in ("", "ctpo_"))
    res["V7_bf16_deterministic"] = bool(rows) and all(r["V7_bf16_repeat_maxabs"] == 0.0 for r in rows)
    res["precision_decision"] = "bf16" if rows and all(r["V4_bf16_ok"] for r in rows) else "fp32"
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "g_validate.json"), "w"), indent=1)
    print(f"[g-validate] V3_pass={res['V3_pass']} V7={res['V7_bf16_deterministic']} "
          f"precision={res['precision_decision']}", flush=True)


if __name__ == "__main__":
    main()
