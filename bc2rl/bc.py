"""`bc2rl bc`: completion-only behaviour cloning with LoRA, resumable, with horizon-stratified order."""
from __future__ import annotations

import json
import os
import random
import sys
import time

import torch

from . import task
from .data import load_data
from .evaluate import run_attempts, summarize
from .modeling import (chat_prompt, load_policy, load_tokenizer, model_info, save_adapter, seq_logprobs,
                       trainable_summary, load_adapter)
from .runtime import (CKPT_INTERVAL_S, CostLedger, Run, StopFlag, atomic_json, atomic_torch_save, rng_state,
                      set_rng_state, write_jsonl_gz)


def stratified_order(problems, seed, epochs):
    """Per epoch: shuffle each horizon, then interleave round-robin so every batch (and the first
    quarter-epoch) holds the horizons in near-equal proportion."""
    by_h = {}
    for i, p in enumerate(problems):
        by_h.setdefault(p.n_inputs, []).append(i)
    order = []
    for e in range(epochs):
        rng = random.Random(seed * 1009 + e)
        lists = {h: rng.sample(ix, len(ix)) for h, ix in sorted(by_h.items())}
        pos = {h: 0 for h in lists}
        while any(pos[h] < len(lists[h]) for h in lists):
            for h in lists:
                if pos[h] < len(lists[h]):
                    order.append(lists[h][pos[h]])
                    pos[h] += 1
    return order


def encode_example(tok, problem, fmt, variant):
    p_ids = tok(chat_prompt(tok, task.prompt_text(problem, variant, fmt)), add_special_tokens=False).input_ids
    c_ids = tok(task.completion_text(problem), add_special_tokens=False).input_ids + [tok.convert_tokens_to_ids("<|im_end|>")]
    return p_ids, c_ids


def bc_loss(model, seqs, pad_id, micro, grad):
    """Token-mean completion cross-entropy over the whole batch (micro-batches accumulate exactly)."""
    n_tok = sum(len(c) for _, c in seqs)
    total = 0.0
    for sl, lp, cm, _ in seq_logprobs(model, seqs, pad_id, micro, grad=grad):
        loss = -(lp * cm).sum() / n_tok
        if grad:
            loss.backward()
        total += float(loss)
    return total, n_tok


def cmd_bc(cfg):
    run = Run(cfg)
    if run.done_ok():
        print(f"[bc] {run.id} already complete")
        return 0
    stop = StopFlag()
    bc, val = cfg["bc"], cfg["val"]
    pd = cfg["proj_data"]
    device = cfg.get("device", "cuda")
    data, man = load_data(cfg, [bc["split"], val["split"]])
    train, vprobs, fmt = data[bc["split"]], data[val["split"]], data["format"][0]
    info = model_info(pd)
    tok = load_tokenizer(info["path"])
    model = load_policy(cfg.get("model_path_override") or info["path"], cfg["model"]["lora"], seed=cfg["seed"], device=device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=bc["lr"], weight_decay=0.0)
    epochs = -(-bc["steps"] * bc["batch"] // len(train))
    order = stratified_order(train, cfg["seed"], epochs)
    enc = [encode_example(tok, p, fmt, cfg["prompt_variant"]) for p in train]
    venc = [encode_example(tok, p, fmt, cfg["prompt_variant"]) for p in vprobs]
    ledger = CostLedger(run.path("cost.json"), "bc")
    state_path = run.path("state", "state.pt")
    step, rows, ckpts = 0, [], []
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        load_adapter(model, st["adapter_path"])
        opt.load_state_dict(st["opt"])
        set_rng_state(st["rng"])
        step, rows, ckpts = st["step"], st["rows"], st["ckpts"]
        print(f"[bc] resumed {run.id} at step {step}")
    atomic_json(run.path("trainable.json"), trainable_summary(model))

    def save_state(note):
        apath = run.path("state", "adapter.safetensors")
        save_adapter(model, apath)
        atomic_torch_save(state_path, {"step": step, "opt": opt.state_dict(), "rng": rng_state(), "rows": rows,
                                       "ckpts": ckpts, "adapter_path": apath, "config_hash": run.chash,
                                       "scheduler": "constant"})
        with open(run.path("metrics.jsonl.tmp"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(run.path("metrics.jsonl.tmp"), run.path("metrics.jsonl"))
        ledger.flush(note=note)

    last_ckpt = time.time()
    model.train()
    while step < bc["steps"]:
        t0 = time.time()
        idx = order[step * bc["batch"]:(step + 1) * bc["batch"]]
        seqs = [enc[i] for i in idx]
        opt.zero_grad(set_to_none=True)
        loss, n_tok = bc_loss(model, seqs, tok.pad_token_id, bc["micro"], grad=True)
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        step += 1
        rows.append({"step": step, "loss": loss, "grad_norm": gn, "tokens": n_tok, "lr": bc["lr"],
                     "horizons": {h: sum(train[i].n_inputs == h for i in idx) for h in (3, 5, 10)},
                     "step_s": time.time() - t0})
        if step in bc["save_steps"]:
            ap = run.path("adapters", f"step{step:04d}.safetensors")
            sha = save_adapter(model, ap)
            with torch.no_grad():
                model.eval()
                hl, _ = bc_loss(model, venc, tok.pad_token_id, bc["micro"], grad=False)
                vrec, gen_s = run_attempts(model, tok, vprobs, fmt, cfg["prompt_variant"], val["attempts"],
                                           val["cap"], seed=cfg["seed"] * 7 + step)
                model.train()
            write_jsonl_gz(run.path("val", f"step{step:04d}.jsonl.gz"), vrec)
            vs = summarize(vrec)
            ckpts.append({"step": step, "epochs": step * bc["batch"] / len(train), "adapter": os.path.relpath(ap, pd),
                          "sha256": sha, "heldout_bc_loss": hl, "val": vs, "val_gen_s": gen_s})
            print(f"[bc] step {step} loss {loss:.4f} heldout {hl:.4f} val {json.dumps(vs['metrics'])}", flush=True)
            save_state(f"step {step}")
            last_ckpt = time.time()
        elif time.time() - last_ckpt > CKPT_INTERVAL_S:
            save_state(f"step {step}")
            last_ckpt = time.time()
        if cfg.get("test_mode") and cfg.get("_debug_kill_after_step") == step:
            os._exit(17)
        if stop.stop:
            save_state(f"signal {stop.signum} at step {step}")
            print(f"[bc] signal {stop.signum}: checkpointed at step {step}; exiting for requeue", flush=True)
            sys.exit(143)
    save_state("complete")
    summary = {"run_id": run.id, "config_hash": run.chash, "steps": step, "checkpoints": ckpts,
               "gpu_s_total": ledger.total_gpu_s(), "segments": ledger.data["segments"],
               "trainable": trainable_summary(model), "data_version": cfg["data"]["version"]}
    atomic_json(run.path("summary.json"), summary)
    atomic_json(run.path("DONE.json"), {"config_hash": run.chash,
                                        "artifacts": [{"path": os.path.relpath(os.path.join(pd, c["adapter"]), run.root),
                                                       "sha256": c["sha256"]} for c in ckpts]})
    res = cfg.get("_results_dir")
    if res:
        atomic_json(os.path.join(res, "summary.json"), summary)
        with open(os.path.join(res, "train_curve.csv"), "w") as f:
            f.write("step,loss,grad_norm,tokens,step_s\n")
            for r in rows:
                f.write(f"{r['step']},{r['loss']:.5f},{r['grad_norm']:.4f},{r['tokens']},{r['step_s']:.3f}\n")
    return 0
