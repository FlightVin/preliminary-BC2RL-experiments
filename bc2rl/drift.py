"""`bc2rl drift`: forward sequence KL(initial || current) on reference-policy responses (A6 stage R1, protocol Section 14.1).

Config key `drift`:
  data_name, select {panel: n}  problems = the n problems of each ext panel with the smallest sha256(salt + ":" + pid)
  salt                          "A6-R drift"
  n_ref, cap, seed, batch_seqs, micro
  out                           directory name under $PROJ_DATA/drift/
  refs   [{tag, adapter, sha256?}]               reference (BC) policies; n_ref responses per problem each
  score  [{ref, tag, adapter, sha256?}]          checkpoints scored on the responses of their reference
For each reference: sample (BF16, evaluation decoding) and record the log-prob of every sampled token during sampling;
score the responses under the reference in FP32 (the check of Section 14.1). For each scored checkpoint: FP32 log-probs
of the same responses. Per response, d = sum over completion tokens (the sampled stop token included; a capped response
has no stop term) of log p_ref - log p_cur. Per problem: KL = mean of d over its responses (failures included);
the per-token diagnostic = sum of d / sum of tokens. A missing adapter file is listed, never re-created.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from . import task
from .data import data_dir
from .evaluate import stable_int
from .modeling import (chat_prompt, completion_logprob_table, decoding_config, eos_ids, load_adapter, load_policy,
                       load_tokenizer, model_info)
from .runtime import CostLedger, atomic_json


def select_problems(probs_by_panel, select, salt):
    out = []
    for panel, n in select.items():
        ps = sorted(probs_by_panel[panel], key=lambda p: hashlib.sha256(f"{salt}:{p.pid}".encode()).hexdigest())
        if len(ps) < n:
            raise SystemExit(f"panel {panel} has {len(ps)} < {n} problems")
        out += ps[:n]
    return out


class _Recorder:
    """Logits processor: keeps the log-softmax of one step and gathers it at the token sampled from it."""

    def __init__(self):
        self.prev, self.steps = None, []

    def __call__(self, input_ids, scores):
        if self.prev is not None:
            self.steps.append(self.prev.gather(1, input_ids[:, -1:]).squeeze(1))
        self.prev = torch.log_softmax(scores.float(), dim=-1)
        return scores

    def finish(self, last_tokens):
        if self.prev is not None:
            self.steps.append(self.prev.gather(1, last_tokens[:, None]).squeeze(1))
        self.prev = None
        return torch.stack(self.steps, 1) if self.steps else None


@torch.no_grad()
def sample_with_logprobs(model, tok, prompt_texts, n_per_prompt, max_new_tokens, seed, batch_seqs=512):
    """As modeling.sample (same seeding), plus `samp_lp`: the sum of the sampling log-probs of the completion tokens."""
    from transformers import LogitsProcessorList
    model.eval()
    tok.padding_side = "left"
    gen_cfg = decoding_config(tok, max_new_tokens)
    stops = set(eos_ids(tok))
    rows = [(i, a) for i in range(len(prompt_texts)) for a in range(n_per_prompt)]
    out = []
    for chunk_i, start in enumerate(range(0, len(rows), batch_seqs)):
        chunk = rows[start:start + batch_seqs]
        enc = tok([prompt_texts[i] for i, _ in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        torch.manual_seed(seed * 100003 + chunk_i)
        rec = _Recorder()
        gen = model.generate(**enc, generation_config=gen_cfg, logits_processor=LogitsProcessorList([rec]))
        plen = enc["input_ids"].shape[1]
        steps = rec.finish(gen[:, -1])
        for r, (i, a) in enumerate(chunk):
            c = gen[r, plen:].tolist()
            finish, cut = "length", len(c)
            for t, tid in enumerate(c):
                if tid in stops:
                    finish, cut = "stop", t + 1
                    break
            c = c[:cut]
            out.append({"prompt_index": i, "attempt": a,
                        "prompt_ids": enc["input_ids"][r][enc["attention_mask"][r].bool()].tolist(),
                        "completion_ids": c, "finish": finish, "n_tokens": len(c),
                        "text": tok.decode([t for t in c if t not in stops], skip_special_tokens=True),
                        "samp_lp": float(steps[r, :cut].sum())})
    return out


def fp32_seq_logprobs(model, seqs, pad_id, cap, micro):
    """Sum of FP32 log-probs of each completion (stop token included). Frozen weights cast exactly; TF32 off."""
    from .online import set_frozen_dtype
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    set_frozen_dtype(model, torch.float32)
    try:
        table, _ = completion_logprob_table(model, seqs, pad_id, cap, micro=micro)
        n = torch.tensor([len(c) for _, c in seqs], device=table.device)
        m = (torch.arange(cap, device=table.device)[None, :] < n[:, None]).float()
        return (table * m).sum(1).double().cpu().numpy()
    finally:
        set_frozen_dtype(model, torch.bfloat16)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32


def per_problem(pids, d, ntok):
    by = {}
    for p, x, n in zip(pids, d, ntok):
        b = by.setdefault(p, [0.0, 0, 0])
        b[0] += float(x)
        b[1] += 1
        b[2] += int(n)
    keys = sorted(by)
    return keys, [by[k][0] / by[k][1] for k in keys], [by[k][0] / max(by[k][2], 1) for k in keys]


def cmd_drift(cfg):
    from .extension import load_ext
    dc = cfg["drift"]
    pd = cfg["proj_data"]
    root = os.path.join(pd, "drift", dc["out"])
    os.makedirs(os.path.join(root, "refs"), exist_ok=True)
    os.makedirs(os.path.join(root, "score"), exist_ok=True)
    res = cfg.get("_results_dir")
    ext, man = load_ext(cfg, dc["data_name"])
    probs = select_problems(ext, dc["select"], dc["salt"])
    fmt = task.load_split(os.path.join(data_dir(cfg), "format.jsonl"))[0]
    info = model_info(pd)
    tok = load_tokenizer(info["path"])
    model = load_policy(cfg.get("model_path_override") or info["path"], cfg["model"]["lora"], seed=0,
                        device=cfg.get("device", "cuda"))
    texts = [chat_prompt(tok, task.prompt_text(p, cfg["prompt_variant"], fmt)) for p in probs]
    ledger = CostLedger(os.path.join(root, f"cost-{cfg['run_id']}.json"), "drift")
    proto = {"data": dc["data_name"], "select": dc["select"], "salt": dc["salt"], "pids": [p.pid for p in probs],
             "n_ref": dc["n_ref"], "cap": dc["cap"], "seed": dc["seed"], "verifier": task.VERIFIER_VERSION}
    missing, done = [], []
    apath = lambda a: a if os.path.isabs(a) else os.path.join(pd, a)

    def load(item):
        if not os.path.exists(apath(item["adapter"])):
            missing.append({"tag": item["tag"], "adapter": item["adapter"]})
            return None
        sha = load_adapter(model, apath(item["adapter"]))
        if item.get("sha256") and item["sha256"] != sha:
            raise SystemExit(f"{item['tag']}: adapter sha256 {sha} != registered {item['sha256']}")
        return sha

    refs = {}
    for r in dc["refs"]:
        path = os.path.join(root, "refs", f"{r['tag']}.pt")
        if os.path.exists(path):
            st = torch.load(path, map_location="cpu", weights_only=False)
            if st["protocol"] == proto:
                refs[r["tag"]] = st
                continue
        sha = load(r)
        if sha is None:
            continue
        seed = dc["seed"] * 1000003 + stable_int(f"ref:{r['tag']}")
        recs = sample_with_logprobs(model, tok, texts, dc["n_ref"], dc["cap"], seed, dc.get("batch_seqs", 512))
        seqs = [(x["prompt_ids"], x["completion_ids"]) for x in recs]
        lp32 = fp32_seq_logprobs(model, seqs, tok.pad_token_id, dc["cap"], dc.get("micro", 8))
        for x, v in zip(recs, lp32):
            p = probs[x["prompt_index"]]
            x["pid"] = p.pid
            x["fp32_lp"] = float(v)
            x["reward"] = task.verify(p.inputs, p.target, x["text"], truncated=(x["finish"] == "length"))["reward"]
        samp = np.array([x["samp_lp"] for x in recs])
        ntok = np.array([x["n_tokens"] for x in recs])
        diff = samp - lp32
        check = {"signed_mean_per_token": float(diff.sum() / ntok.sum()), "median_abs_seq": float(np.median(np.abs(diff))),
                 "max_abs_seq": float(np.abs(diff).max())}
        check["pass"] = abs(check["signed_mean_per_token"]) <= 0.005
        st = {"protocol": proto, "tag": r["tag"], "adapter": r["adapter"], "adapter_sha256": sha, "eval_seed": seed,
              "records": recs, "check": check,
              "summary": {"responses": len(recs), "success": float(np.mean([x["reward"] for x in recs])),
                          "capped": int(sum(x["finish"] == "length" for x in recs)), "mean_tokens": float(ntok.mean())}}
        torch.save(st, path + ".tmp")
        os.replace(path + ".tmp", path)
        refs[r["tag"]] = st
        ledger.flush(note=f"ref {r['tag']}")
        print(f"[drift] ref {r['tag']}: {len(recs)} responses success={st['summary']['success']:.3f} "
              f"check={json.dumps(check)}", flush=True)
        if res:
            atomic_json(os.path.join(res, f"drift_ref_{r['tag']}.json"),
                        {k: st[k] for k in ("tag", "adapter", "adapter_sha256", "check", "summary")})
    for it in dc["score"]:
        ref = refs.get(it["ref"])
        path = os.path.join(root, "score", f"{it['tag']}.json")
        if ref is None:
            missing.append({"tag": it["tag"], "adapter": it["adapter"], "reason": f"reference {it['ref']} missing"})
            continue
        if os.path.exists(path):
            old = json.load(open(path))
            if old["ref_sha256"] == ref["adapter_sha256"] and old["protocol"] == proto and \
                    (not it.get("sha256") or old["adapter_sha256"] == it["sha256"]) and \
                    old["adapter"] == it["adapter"]:
                done.append(it["tag"])
                if res:
                    atomic_json(os.path.join(res, f"drift_{it['tag']}.json"), {k: v for k, v in old.items() if k != "protocol"})
                continue
        sha = load(it)
        if sha is None:
            continue
        recs = ref["records"]
        seqs = [(x["prompt_ids"], x["completion_ids"]) for x in recs]
        cur = fp32_seq_logprobs(model, seqs, tok.pad_token_id, dc["cap"], dc.get("micro", 8))
        d = np.array([x["fp32_lp"] for x in recs]) - cur
        ntok = [x["n_tokens"] for x in recs]
        pids, kl, kl_tok = per_problem([x["pid"] for x in recs], d, ntok)
        out = {"tag": it["tag"], "ref": it["ref"], "adapter": it["adapter"], "adapter_sha256": sha,
               "ref_sha256": ref["adapter_sha256"], "protocol": proto, "kl_mean": float(np.mean(kl)),
               "kl_tok_mean": float(d.sum() / sum(ntok)), "per_problem": {"pid": pids, "kl": kl, "kl_tok": kl_tok}}
        atomic_json(path, out)
        ledger.flush(note=f"score {it['tag']}")
        done.append(it["tag"])
        print(f"[drift] {it['tag']} (ref {it['ref']}): KL_seq={out['kl_mean']:.4f} per-token={out['kl_tok_mean']:.5f}",
              flush=True)
        if res:
            atomic_json(os.path.join(res, f"drift_{it['tag']}.json"), {k: v for k, v in out.items() if k != "protocol"})
    ledger.flush(note="complete")
    if res:
        atomic_json(os.path.join(res, "drift_run.json"), {"run_id": cfg["run_id"], "done": done, "missing": missing,
                                                          "gpu_s": ledger.total_gpu_s(), "pids": proto["pids"]})
    print(f"[drift] done={len(done)} missing={len(missing)} {json.dumps(missing)[:500]}", flush=True)
    return 0
