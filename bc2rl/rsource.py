"""A6 stage R3 (protocol/AMENDMENT-A6-2026-10-26.md Sections 14.3 and 14.9): the response-source experiment.

`bc2rl rprep`   common pool and initial-policy caches.
  Screening: candidates of ext panel `rprep.panel` in draw order, in chunks; every screening checkpoint samples
  `screen.attempts` independent attempts per candidate. A candidate qualifies when every screening checkpoint solves it
  at least once. Screening stops after the first chunk that brings the qualifying count to `screen.target`, or at
  `screen.max` candidates. Pool = the first `target` qualifying candidates in draw order; if fewer qualify, the first
  `fallback`; fewer than `fallback` is a failed gate (exit 3).
  Caches (arm I): for each cache checkpoint and pool problem, sample until the first verified success (`first_success`),
  at most `cache.attempts` attempts, seeds independent of screening.
  Writes $PROJ_DATA/rsource/<out>/{screen.json, pool.json, caches/<tag>.json} and a compact distillate.

`bc2rl rsource` one four-arm block from one starting checkpoint (one GPU, one base model, four adapter copies, four
  AdamW states). Arms: S = the generator solver trace (`task.completion_text` + <|im_end|>), I = the cached initial-policy
  response, F = a fresh verified response of the frozen starting policy, O = a fresh verified response of the arm-O
  policy. Each round: draw 32 prompt occurrences (same stream in all arms); sample the F and O targets (F first, then O,
  both before any update); m_i = 0 when F or O has no verified target within the attempt cap, in all four arms; then each
  arm takes one AdamW step on
      L(theta) = -(1 / (P x cap)) * sum_i m_i * sum_{t=1..T_i} log pi_theta(y_it | x_i, y_i<t)
  (owner definition, protocol 14.9.1; P x cap = 32 x 256, fixed; T_i counts response tokens incl. the sampled stop token;
  teacher forcing; targets and masks are fixed data). Gradient-norm clip 1.0, AdamW lr 1e-5, weight decay 0, constant
  learning rate. A round with every occurrence masked takes no optimizer step in any arm; it is counted.
  A verified target ends with a sampled stop token (`finish == "stop"`): a truncated response is never a target.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

from . import task
from .data import data_dir
from .evaluate import run_attempts, stable_int
from .extension import load_ext
from .losses import filtered_loss_sum
from .modeling import (chat_prompt, eos_ids, file_sha256, gather_completion, load_adapter, load_policy, load_tokenizer,
                       model_info, sample, save_adapter, seq_logprobs)
from .online import prompt_stream
from .runtime import CKPT_INTERVAL_S, CostLedger, Run, StopFlag, atomic_json, atomic_torch_save, rng_state, \
    set_rng_state, write_jsonl_gz

ARMS = ("S", "I", "F", "O")


def _abs(pd, p):
    return p if os.path.isabs(p) else os.path.join(pd, p)


def _success(cfg, p, o, key):
    """Verified success: verifier reward 1 and a sampled stop token. Test mode may replace the verifier."""
    dbg = cfg.get("test_mode") and cfg.get("_debug_verify")
    if dbg == "random":
        return stable_int(key) % 2 == 0
    if dbg == "all":
        return True
    if o["finish"] != "stop":
        return False
    return task.verify(p.inputs, p.target, o["text"], truncated=False)["reward"] == 1


def first_success(cfg, model, tok, probs, texts, max_attempts, batch, seed, tag=""):
    """Per problem: the first verified success in attempt order, sampling `batch` attempts at a time, at most
    `max_attempts`. Returns (targets, cost): targets[i] = {"ids", "text", "attempt"} or None; every generated attempt
    is counted (also those after the first success within its batch)."""
    n = len(probs)
    out, pending = [None] * n, list(range(n))
    cost = {"attempts": 0, "tokens": 0, "first_index_sum": 0, "found": 0}
    for b in range(0, max_attempts, batch):
        if not pending:
            break
        k = min(batch, max_attempts - b)
        outs = sample(model, tok, [texts[i] for i in pending], k, cfg["_cap"], seed=seed * 1009 + b // batch,
                      batch_seqs=cfg.get("_batch_seqs", 512))
        cost["attempts"] += len(outs)
        cost["tokens"] += sum(o["n_tokens"] for o in outs)
        done = set()
        for o in outs:
            i = pending[o["prompt_index"]]
            if i in done:
                continue
            if _success(cfg, probs[i], o, f"{tag}:{probs[i].pid}:{b + o['attempt']}:{seed}"):
                out[i] = {"ids": o["completion_ids"], "text": o["text"], "attempt": b + o["attempt"],
                          "finish": o["finish"]}
                done.add(i)
                cost["found"] += 1
                cost["first_index_sum"] += b + o["attempt"]
        pending = [i for i in pending if i not in done]
    return out, cost


def _model(cfg):
    pd = cfg["proj_data"]
    info = model_info(pd)
    tok = load_tokenizer(info["path"])
    model = load_policy(cfg.get("model_path_override") or info["path"], cfg["model"]["lora"], seed=cfg.get("seed", 0),
                        device=cfg.get("device", "cuda"))
    return tok, model


def _load(model, pd, ck):
    sha = load_adapter(model, _abs(pd, ck["adapter"]))
    if ck.get("sha256") and ck["sha256"] != sha:
        raise SystemExit(f"{ck['tag']}: adapter sha256 {sha} != registered {ck['sha256']}")
    return sha


def cmd_rprep(cfg):
    rp, pd = cfg["rprep"], cfg["proj_data"]
    cfg["_cap"], cfg["_batch_seqs"] = rp["cap"], rp.get("batch_seqs", 512)
    out = os.path.join(pd, "rsource", rp["out"])
    os.makedirs(os.path.join(out, "caches"), exist_ok=True)
    ext, man = load_ext(cfg, rp["data_name"])
    cands = ext[rp["panel"]]
    fmt = task.load_split(os.path.join(data_dir(cfg), "format.jsonl"))[0]
    tok, model = _model(cfg)
    ledger = CostLedger(os.path.join(out, f"cost-{cfg['run_id']}.json"), "rprep")
    sc = rp["screen"]
    scr_path = os.path.join(out, "screen.json")
    scr = json.load(open(scr_path)) if os.path.exists(scr_path) else {"chunks": [], "counts": {}, "cost": {}}
    proto = {k: rp[k] for k in ("data_name", "panel", "cap", "screen")}
    proto.update(panel_sha256=man["panels"][rp["panel"]]["sha256"],
                 checkpoints=[[c["tag"], c["adapter"]] for c in rp["screen_checkpoints"]])
    if scr.get("protocol") not in (None, proto):
        raise SystemExit("screen.json was written under a different protocol")
    scr["protocol"] = proto
    stop = StopFlag()

    def qualified(n_done):
        return [p.pid for p in cands[:n_done]
                if all(scr["counts"][c["tag"]].get(p.pid, 0) > 0 for c in rp["screen_checkpoints"])]

    n_done = sum(scr["chunks"])
    while len(qualified(n_done)) < sc["target"] and n_done < min(sc["max"], len(cands)):
        chunk = cands[n_done:min(n_done + sc["chunk"], sc["max"], len(cands))]
        for ck in rp["screen_checkpoints"]:
            key = f"{ck['tag']}@{n_done}"
            if key in scr["cost"]:
                continue
            _load(model, pd, ck)
            recs, gen_s = run_attempts(model, tok, chunk, fmt, cfg["prompt_variant"], sc["attempts"], rp["cap"],
                                       seed=sc["seed"] * 1000003 + stable_int(key), batch_seqs=cfg["_batch_seqs"])
            cnt = scr["counts"].setdefault(ck["tag"], {})
            for p in chunk:
                cnt[p.pid] = 0
            for r in recs:
                ok = _success(cfg, None, {"finish": "stop"}, f"scr:{key}:{r['pid']}:{r['attempt']}") \
                    if cfg.get("test_mode") and cfg.get("_debug_verify") else (r["reward"] == 1)
                cnt[r["pid"]] += int(ok)
            scr["cost"][key] = {"attempts": len(recs), "tokens": int(sum(r["n_tokens"] for r in recs)), "gen_s": gen_s}
            atomic_json(scr_path, scr)
            ledger.flush(note=f"screen {key}")
            if stop.stop:
                sys.exit(143)
        n_done += len(chunk)
        scr["chunks"].append(len(chunk))
        atomic_json(scr_path, scr)
        print(f"[rprep] screened {n_done} candidates; qualified {len(qualified(n_done))}", flush=True)
    q = qualified(n_done)
    if len(q) >= sc["target"]:
        pool, rule = q[:sc["target"]], "target"
    elif len(q) >= sc["fallback"]:
        pool, rule = q[:sc["fallback"]], "fallback"
    else:
        atomic_json(os.path.join(out, "pool.json"), {"pool": [], "rule": "failed", "qualified": len(q), "screened": n_done})
        print(f"[rprep] GATE FAILED: {len(q)} qualifying < fallback {sc['fallback']}", flush=True)
        return 3
    pool_info = {"pool": pool, "rule": rule, "qualified": len(q), "screened": n_done,
                 "excluded_not_qualified": n_done - len(q), "protocol": proto}
    atomic_json(os.path.join(out, "pool.json"), pool_info)
    by_pid = {p.pid: p for p in cands}
    pprobs = [by_pid[x] for x in pool]
    texts = [chat_prompt(tok, task.prompt_text(p, cfg["prompt_variant"], fmt)) for p in pprobs]
    ca = rp["cache"]
    summ = {}
    for ck in rp["cache_checkpoints"]:
        cpath = os.path.join(out, "caches", f"{ck['tag']}.json")
        if os.path.exists(cpath):
            summ[ck["tag"]] = json.load(open(cpath))["summary"]
            continue
        sha = _load(model, pd, ck)
        t0 = time.time()
        tg, cost = first_success(cfg, model, tok, pprobs, texts, ca["attempts"], ca["batch"],
                                 seed=ca["seed"] * 1000003 + stable_int(f"cache:{ck['tag']}"), tag=f"cache:{ck['tag']}")
        s = {"found": cost["found"], "missing": len(pool) - cost["found"], "attempts": cost["attempts"],
             "tokens": cost["tokens"], "mean_first_index": cost["first_index_sum"] / max(cost["found"], 1),
             "gen_s": time.time() - t0}
        atomic_json(cpath, {"tag": ck["tag"], "adapter": ck["adapter"], "adapter_sha256": sha, "protocol": ca,
                            "targets": {p.pid: t for p, t in zip(pprobs, tg)}, "summary": s})
        summ[ck["tag"]] = s
        ledger.flush(note=f"cache {ck['tag']}")
        print(f"[rprep] cache {ck['tag']}: found {s['found']}/{len(pool)} attempts {s['attempts']}", flush=True)
        if stop.stop:
            sys.exit(143)
    conf = [c["tag"] for c in rp["cache_checkpoints"] if c.get("role", "conf") == "conf"]
    caches = {t: json.load(open(os.path.join(out, "caches", f"{t}.json")))["targets"] for t in summ}
    common = [x for x in pool if all(caches[t].get(x) for t in conf)]
    res = {"pool_rule": rule, "pool": len(pool), "qualified": len(q), "screened": n_done,
           "screen_attempts": sum(v["attempts"] for v in scr["cost"].values()),
           "screen_tokens": sum(v["tokens"] for v in scr["cost"].values()), "caches": summ,
           "common_pool": len(common), "excluded_no_conf_cache": len(pool) - len(common),
           "gpu_s": ledger.total_gpu_s()}
    atomic_json(os.path.join(out, "common_pool.json"), {"pool": common, "conf_checkpoints": conf, "summary": res})
    ledger.flush(note="complete")
    if cfg.get("_results_dir"):
        atomic_json(os.path.join(cfg["_results_dir"], "rprep_summary.json"), res)
    print(f"[rprep] done: {json.dumps({k: v for k, v in res.items() if k != 'caches'})}", flush=True)
    return 0


def sup_loss_sum(lp_g, tmask):
    """Summed token negative log-likelihood of the accepted rows (the numerator of L; divide by P x cap outside)."""
    return filtered_loss_sum(lp_g, tmask)


def cmd_rsource(cfg):
    run = Run(cfg)
    if run.done_ok():
        print(f"[rsource] {run.id} already complete")
        return 0
    rs, pd = cfg["rsource"], cfg["proj_data"]
    P, R, cap, lr = rs["prompts_per_round"], rs["rounds"], rs["cap"], rs["lr"]
    cfg["_cap"], cfg["_batch_seqs"] = cap, rs.get("batch_seqs", 512)
    den = float(P * cap)
    prep = os.path.join(pd, "rsource", rs["prep_out"])
    common = json.load(open(os.path.join(prep, "common_pool.json")))["pool"]
    cache = json.load(open(os.path.join(prep, "caches", f"{rs['cache_tag']}.json")))
    if rs["init"].get("sha256") and cache["adapter_sha256"] != rs["init"]["sha256"]:
        raise SystemExit("cache checkpoint differs from the block's starting checkpoint")
    pool_ids = [x for x in common if cache["targets"].get(x)]
    ext, _ = load_ext(cfg, rs["data_name"])
    by_pid = {p.pid: p for p in ext[rs["panel"]]}
    pool = [by_pid[x] for x in pool_ids]
    fmt = task.load_split(os.path.join(data_dir(cfg), "format.jsonl"))[0]
    tok, model = _model(cfg)
    init_sha = _load(model, pd, rs["init"])
    if init_sha != cache["adapter_sha256"]:
        raise SystemExit("cache was built from a different adapter file")
    stops = set(eos_ids(tok))
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    texts = [chat_prompt(tok, task.prompt_text(p, cfg["prompt_variant"], fmt)) for p in pool]
    p_ids = [tok(t, add_special_tokens=False).input_ids for t in texts]
    s_ids = [tok(task.completion_text(p), add_special_tokens=False).input_ids + [im_end] for p in pool]
    i_ids = [cache["targets"][p.pid]["ids"] for p in pool]
    for ids in s_ids + i_ids:
        if len(ids) > cap or ids[-1] not in stops:
            raise SystemExit("a fixed target exceeds the cap or does not end with a stop token")
    stream = prompt_stream(len(pool), P, R, rs["stream_seed"])
    params = [p for p in model.parameters() if p.requires_grad]
    init_w = [p.detach().clone() for p in params]
    arm_w = {a: [torch.nn.Parameter(p.detach().clone()) for p in params] for a in ARMS}
    opts = {a: torch.optim.AdamW(arm_w[a], lr=lr, weight_decay=0.0) for a in ARMS}   # fresh state at the branch

    def use(ws):
        for p, w in zip(params, ws):
            p.data = w.data            # the model computes with this arm's tensors (shared storage)

    ledger = CostLedger(run.path("cost.json"), "rsource")
    state_path = run.path("state", "state.pt")
    rnd, rows, ckpts = 0, [], []
    counters = {"opt_steps": {a: 0 for a in ARMS}, "skipped_rounds": 0, "masked": 0, "occurrences": 0,
                "fresh_attempts": {"F": 0, "O": 0}, "fresh_tokens": {"F": 0, "O": 0},
                "target_tokens": {a: 0 for a in ARMS}}
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for a in ARMS:
                for w, v in zip(arm_w[a], st["arm_w"][a]):
                    w.copy_(v.to(w.device))
                opts[a].load_state_dict(st["opt"][a])
        set_rng_state(st["rng"])
        rnd, rows, ckpts, counters = st["round"], st["rows"], st["ckpts"], st["counters"]
        print(f"[rsource] resumed {run.id} at round {rnd}", flush=True)
    else:
        atomic_json(run.path("init.json"), {"init": rs["init"], "init_sha256": init_sha, "pool": len(pool),
                                            "pool_excluded_no_cache": len(common) - len(pool)})

    def save_state(note):
        atomic_torch_save(state_path, {"round": rnd, "arm_w": {a: [w.detach().cpu() for w in arm_w[a]] for a in ARMS},
                                       "opt": {a: opts[a].state_dict() for a in ARMS}, "rng": rng_state(),
                                       "rows": rows, "ckpts": ckpts, "counters": counters, "config_hash": run.chash})
        with open(run.path("metrics.jsonl.tmp"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(run.path("metrics.jsonl.tmp"), run.path("metrics.jsonl"))
        ledger.flush(note=note)

    def save_ckpt(r_now):
        entry = {"round": r_now, "adapters": {}}
        for a in ARMS:
            use(arm_w[a])
            ap = run.path("adapters", a, f"round{r_now:04d}.safetensors")
            entry["adapters"][a] = {"path": os.path.relpath(ap, pd), "sha256": save_adapter(model, ap)}
        use(init_w)
        return entry

    stop = StopFlag()
    dbg_fail = {(int(r), k): set(v) for r, k, v in (cfg.get("_debug_fail") or [])} if cfg.get("test_mode") else {}
    if rnd == 0 and 0 in rs["save_rounds"] and not any(c["round"] == 0 for c in ckpts):
        ckpts.append(save_ckpt(0))
        save_state("round 0")
    last_ckpt = time.time()
    fa, fb = rs["fresh"]["attempts"], rs["fresh"]["batch"]
    while rnd < R:
        t0 = time.time()
        occ = stream[rnd]
        probs = [pool[i] for i in occ]
        tg, cost = {}, {}
        for a, ws in (("F", init_w), ("O", arm_w["O"])):   # both before any update of this round
            use(ws)
            tg[a], cost[a] = first_success(cfg, model, tok, probs, [texts[i] for i in occ], fa, fb,
                                           seed=rs["sample_seed"] * 1000003 + 2 * rnd + (a == "O"), tag=f"{a}:{rnd}")
            for k, v in enumerate(occ):
                if a in dbg_fail.get((rnd + 1, k), ()):
                    tg[a][k] = None
            counters["fresh_attempts"][a] += cost[a]["attempts"]
            counters["fresh_tokens"][a] += cost[a]["tokens"]
        use(init_w)
        accept = [k for k in range(P) if tg["F"][k] is not None and tg["O"][k] is not None]
        targets = {"S": [s_ids[i] for i in occ], "I": [i_ids[i] for i in occ],
                   "F": [t["ids"] if t else None for t in tg["F"]], "O": [t["ids"] if t else None for t in tg["O"]]}
        n_mask = P - len(accept)
        counters["masked"] += n_mask
        counters["occurrences"] += P
        row = {"round": rnd + 1, "masked": n_mask, "accepted": len(accept),
               "fresh": {a: {"attempts": cost[a]["attempts"], "tokens": cost[a]["tokens"], "found": cost[a]["found"]}
                         for a in ("F", "O")}, "arms": {}}
        if not accept:
            counters["skipped_rounds"] += 1          # no optimizer (or scheduler) step in any arm; the round counts
        else:
            for a in ARMS:
                use(arm_w[a])
                model.train()
                opts[a].zero_grad(set_to_none=True)
                for p in params:
                    p.grad = None
                seqs = [(p_ids[occ[k]], targets[a][k]) for k in accept]
                ltot, ntok = 0.0, 0
                for sl, lp, cm, _ in seq_logprobs(model, seqs, tok.pad_token_id, rs["micro"], grad=True):
                    lp_g = gather_completion(lp, cm, cap)
                    lens = torch.tensor([len(c) for _, c in seqs[sl]], device=lp_g.device)
                    m = (torch.arange(cap, device=lp_g.device).unsqueeze(0) < lens.unsqueeze(1)).float()
                    loss = sup_loss_sum(lp_g, m) / den
                    loss.backward()
                    ltot += float(loss)
                    ntok += int(lens.sum())
                for p, w in zip(params, arm_w[a]):
                    w.grad, p.grad = p.grad, None
                gn = float(torch.nn.utils.clip_grad_norm_(arm_w[a], 1.0))
                if not np.isfinite(gn) or not np.isfinite(ltot):
                    raise SystemExit(f"non-finite loss or gradient in arm {a} at round {rnd + 1}")
                opts[a].step()
                opts[a].zero_grad(set_to_none=True)
                counters["opt_steps"][a] += 1
                counters["target_tokens"][a] += ntok
                row["arms"][a] = {"loss": ltot, "grad_norm": gn, "tokens": ntok}
            use(init_w)
        rnd += 1
        row["round_s"] = time.time() - t0
        rows.append(row)
        write_jsonl_gz(run.path("records", f"round{rnd:04d}.jsonl.gz"),
                       [{"round": rnd, "k": k, "pid": probs[k].pid, "accepted": k in accept,
                         **{f"{a}_attempt": (tg[a][k]["attempt"] if tg[a][k] else None) for a in ("F", "O")},
                         **{f"{a}_text": (tg[a][k]["text"] if tg[a][k] else None) for a in ("F", "O")}}
                        for k in range(P)])
        if rnd in rs["save_rounds"]:
            ckpts = [c for c in ckpts if c["round"] != rnd] + [dict(save_ckpt(rnd), counters=json.loads(json.dumps(counters)),
                                                                    gpu_s_so_far=ledger.total_gpu_s())]
            save_state(f"round {rnd}")
            last_ckpt = time.time()
        elif time.time() - last_ckpt > CKPT_INTERVAL_S or (cfg.get("test_mode") and
                                                            rnd % cfg.get("_debug_ckpt_every", 10 ** 9) == 0):
            save_state(f"round {rnd}")
            last_ckpt = time.time()
        if rnd % 8 == 0 or rnd == 1:
            print(f"[rsource] r{rnd} masked={n_mask} F_att={cost['F']['attempts']} O_att={cost['O']['attempts']} "
                  + " ".join(f"{a}:loss={v['loss']:.4f},gn={v['grad_norm']:.3f}" for a, v in row["arms"].items())
                  + f" {row['round_s']:.1f}s", flush=True)
        if cfg.get("test_mode") and cfg.get("_debug_kill_after_round") == rnd:
            os._exit(17)
        if stop.stop:
            save_state(f"signal {stop.signum} at round {rnd}")
            sys.exit(143)
    save_state("complete")
    frac = counters["masked"] / max(counters["occurrences"], 1)
    summary = {"run_id": run.id, "config_hash": run.chash, "init": rs["init"], "init_sha256": init_sha,
               "rounds": rnd, "pool": len(pool), "counters": counters, "masked_frac": frac,
               "exposure_matching_failed": frac > rs.get("mask_limit", 0.05), "checkpoints": ckpts,
               "loss": "L = -(1/(P*cap)) sum_i m_i sum_t log p(y_it); P*cap fixed", "denominator": den, "lr": lr,
               "gpu_s_total": ledger.total_gpu_s(), "segments": ledger.data["segments"]}
    atomic_json(run.path("summary.json"), summary)
    atomic_json(run.path("DONE.json"), {"config_hash": run.chash, "artifacts": [
        {"path": os.path.relpath(_abs(pd, v["path"]), run.root), "sha256": v["sha256"]}
        for c in ckpts for v in c["adapters"].values()]})
    res = cfg.get("_results_dir")
    if res:
        atomic_json(os.path.join(res, "summary.json"), {k: v for k, v in summary.items() if k != "segments"})
        with open(os.path.join(res, "rounds.csv"), "w") as f:
            f.write("round,masked,F_attempts,O_attempts,F_tokens,O_tokens,round_s," +
                    ",".join(f"{a}_loss,{a}_gn,{a}_tokens" for a in ARMS) + "\n")
            for r in rows:
                arms = ",".join(",".join(str(r["arms"].get(a, {}).get(k, "")) for k in ("loss", "grad_norm", "tokens"))
                                for a in ARMS)
                f.write(f"{r['round']},{r['masked']},{r['fresh']['F']['attempts']},{r['fresh']['O']['attempts']},"
                        f"{r['fresh']['F']['tokens']},{r['fresh']['O']['tokens']},{r['round_s']:.2f},{arms}\n")
    print(f"[rsource] done {run.id}: masked {frac:.4f} steps {counters['opt_steps']}", flush=True)
    return 0
