"""Repeated-sampling evaluation (`bc2rl evaluate`) and the shared attempt/summary helpers."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np
import torch

from . import metrics, task
from .data import load_data
from .modeling import (DECODING_RECORD, chat_prompt, load_adapter, load_policy, load_tokenizer,
                       model_info, sample)
from .runtime import CostLedger, atomic_json, read_jsonl_gz, write_jsonl_gz

FREEZE_FILE = ("frozen", "protocol_frozen.json")


def stable_int(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def run_attempts(model, tok, problems, fmt, variant, n, cap, seed, batch_seqs=512, adapter_off=False):
    texts = [chat_prompt(tok, task.prompt_text(p, variant, fmt)) for p in problems]
    t0 = time.time()
    outs = sample(model, tok, texts, n, cap, seed, batch_seqs=batch_seqs, adapter_off=adapter_off)
    gen_s = time.time() - t0
    recs = []
    for o in outs:
        p = problems[o["prompt_index"]]
        v = task.verify(p.inputs, p.target, o["text"], truncated=(o["finish"] == "length"))
        recs.append({"pid": p.pid, "n_inputs": p.n_inputs, "attempt": o["attempt"], "text": o["text"],
                     "finish": o["finish"], "n_tokens": o["n_tokens"], "prompt_tokens": len(o["prompt_ids"]),
                     **v})
    return recs, gen_s


def counts_by_problem(recs):
    c, n, h = defaultdict(int), defaultdict(int), {}
    for r in recs:
        c[r["pid"]] += r["reward"]
        n[r["pid"]] += 1
        h[r["pid"]] = r["n_inputs"]
    pids = sorted(n)
    return pids, np.array([n[p] for p in pids]), np.array([c[p] for p in pids]), np.array([h[p] for p in pids])


def summarize(recs):
    pids, n, c, h = counts_by_problem(recs)
    out = {"n_problems": len(pids), "attempts_per_problem": sorted(set(n.tolist())), "metrics": {}, "by_inputs": {}}
    nmin = int(n.min())
    for m in ("pass@1", "pass@4", "pass^4", "pass^2", "mixed4"):
        if m != "pass@1" and nmin < 4 and m != "pass^2":
            continue
        if m == "pass^2" and nmin < 2:
            continue
        vals = metrics.per_problem(n, c, m)
        overall, byh = metrics.horizon_weighted(vals, h)
        out["metrics"][m] = overall
        for hh, v in byh.items():
            out["by_inputs"].setdefault(str(hh), {})[m] = v
    for hh in sorted(set(h.tolist())):
        rs = [r for r in recs if r["n_inputs"] == hh]
        cc = c[h == hh]
        out["by_inputs"].setdefault(str(hh), {}).update({
            "format_valid": float(np.mean([r["format_valid"] for r in rs])),
            "truncated": float(np.mean([r["finish"] == "length" for r in rs])),
            "mean_tokens": float(np.mean([r["n_tokens"] for r in rs])),
            "max_tokens": int(max(r["n_tokens"] for r in rs)),
            "mean_valid_prefix": float(np.mean([r["valid_prefix"] for r in rs])),
            "errors": dict(Counter(r["error"] for r in rs)),
            "problems_with_any_success": int((cc > 0).sum()),
            "problems_all_success": int((cc == n[h == hh]).sum()),
            "success_count_hist": dict(Counter(int(x) for x in cc)),
        })
    return out


def write_counts_csv(path, recs, tag):
    pids, n, c, h = counts_by_problem(recs)
    tok = defaultdict(list)
    for r in recs:
        tok[r["pid"]].append(r["n_tokens"])
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "pid", "n_inputs", "n", "c", "mean_tokens"])
        for p, nn, cc, hh in zip(pids, n, c, h):
            w.writerow([tag, p, int(hh), int(nn), int(cc), round(float(np.mean(tok[p])), 1)])


def protocol_record(cfg, ev):
    return {"split": ev["split"], "variant": ev["variant"], "attempts": ev["attempts"], "cap": ev["cap"],
            "decoding": DECODING_RECORD, "verifier": task.VERIFIER_VERSION, "data_version": cfg["data"]["version"],
            "seed": ev["seed"]}


def cmd_evaluate(cfg):
    ev = cfg["evaluate"]
    pd = cfg["proj_data"]
    if ev["split"] == "test":
        if not ev.get("allow_test") or not os.path.exists(os.path.join(pd, *FREEZE_FILE)):
            raise SystemExit("test split is locked until the protocol freeze file exists and allow_test is set")
        fz = json.load(open(os.path.join(pd, *FREEZE_FILE)))
        if fz.get("lr") is None or fz.get("rounds") is None or fz.get("cap") != ev["cap"]:
            raise SystemExit(f"freeze file incomplete or cap mismatch: {fz}")
    data, man = load_data(cfg, [ev["split"]])
    problems = data[ev["split"]]
    if ev.get("limit_per_inputs"):
        keep = Counter()
        sel = []
        for p in problems:
            if keep[p.n_inputs] < ev["limit_per_inputs"]:
                keep[p.n_inputs] += 1
                sel.append(p)
        problems = sel
    fmt = data["format"][0]
    info = model_info(pd)
    tok = load_tokenizer(info["path"])
    model = load_policy(info["path"], cfg["model"]["lora"], seed=0)
    out_dir = os.path.join(pd, "evals", cfg["run_id"])
    os.makedirs(out_dir, exist_ok=True)
    ledger = CostLedger(os.path.join(out_dir, "cost.json"), "evaluate")
    res_dir = cfg.get("_results_dir")
    proto = protocol_record(cfg, ev)
    summaries = {}
    for ck in ev["checkpoints"]:
        tag, adapter = ck["tag"], ck["adapter"]
        rec_path = os.path.join(out_dir, f"{tag}.jsonl.gz")
        sum_path = os.path.join(out_dir, f"{tag}.summary.json")
        if adapter == "base":
            asha, off = "base", True
        else:
            apath = adapter if os.path.isabs(adapter) else os.path.join(pd, adapter)
            asha, off = load_adapter(model, apath), False
        if os.path.exists(sum_path):
            old = json.load(open(sum_path))
            if old.get("adapter_sha256") == asha and old.get("protocol") == proto and os.path.exists(rec_path):
                summaries[tag] = old
                print(f"[eval] {tag}: reuse (same adapter hash and protocol)")
                continue
        seed = ev["seed"] * 1000003 + stable_int(tag)
        torch.cuda.synchronize()
        t0 = time.time()
        recs, gen_s = run_attempts(model, tok, problems, fmt, ev["variant"], ev["attempts"], ev["cap"], seed,
                                   ev.get("batch_seqs", 512), adapter_off=off)
        wall = time.time() - t0
        for r in recs:
            r.update(tag=tag, adapter_sha256=asha, eval_seed=seed, variant=ev["variant"], cap=ev["cap"])
        write_jsonl_gz(rec_path, recs)
        s = summarize(recs)
        s.update(tag=tag, adapter=adapter, adapter_sha256=asha, protocol=proto, records=rec_path,
                 wall_s=wall, gen_s=gen_s, completions=len(recs),
                 generated_tokens=int(sum(r["n_tokens"] for r in recs)),
                 gpu=torch.cuda.get_device_name(0))
        atomic_json(sum_path, s)
        summaries[tag] = s
        ledger.flush(note=f"done {tag}")
        shown = "metrics hidden (test split; read only through the frozen report)" if ev["split"] == "test" \
            else json.dumps(s["metrics"])
        print(f"[eval] {tag}: {shown} wall={wall:.0f}s tokens={s['generated_tokens']}")
        if res_dir:
            atomic_json(os.path.join(res_dir, f"eval_{tag}.json"),
                        {k: v for k, v in s.items() if k != "records"})
            write_counts_csv(os.path.join(res_dir, "counts.csv"), recs, tag)
            if ck.get("save_samples"):
                sample_rows = [r for r in recs if r["attempt"] == 0][:60]
                write_jsonl_gz(os.path.join(res_dir, f"samples_{tag}.jsonl.gz"), sample_rows)
    ledger.flush(note="complete")
    atomic_json(os.path.join(out_dir, "DONE.json"), {"tags": sorted(summaries), "protocol": proto})
    if res_dir:
        atomic_json(os.path.join(res_dir, "summary.json"),
                    {"run_id": cfg["run_id"], "protocol": proto, "gpu_s": ledger.total_gpu_s(),
                     "metrics": {t: s["metrics"] for t, s in summaries.items()}})
    return 0


def load_eval_records(proj_data, eval_id, tag):
    return read_jsonl_gz(os.path.join(proj_data, "evals", eval_id, f"{tag}.jsonl.gz"))
