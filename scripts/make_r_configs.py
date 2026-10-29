#!/usr/bin/env python3
"""Write the stage-R configurations and worklists (protocol/AMENDMENT-A6-2026-10-26.md Section 14). Stdlib only.

  drift (14.1)  configs/r/drift-s<seed>.json   run bc2rl-r1-drift-s<seed>   worklist r1-drift (one task per seed)
                references: the BC adapters of steps 16, 32 (re-run), 64, 128 of the seed;
                scored: rounds 64, 128, 256 of every A5/A6 branch of the seed (M GRPO and filtered, Y3 MASK and T3, Y4 steps
                32 and 64), each against its own BC reference. Round 0 is the reference itself (KL 0 by definition).
  rdata (14.9.2) configs/r/xdata-rs1.json      ext data rs1 (seed 20261030; every earlier ext dataset excluded):
                rsc 2,048 screening candidates, rd3 256 (dev panel), rf3 768 and rf5 128 (confirmation panels)
  rdev (14.9.2) configs/r/rprep.json, rs-dev-<init>-s7.json, xeval-rd3-dev-<init>-s7.json, xeval-yd3-dev-<init>-s7.json
                worklist r3-dev: prep (screening on the 6 confirmation checkpoints, I caches for those 6 and the 2 dev
                checkpoints), then the two seed-7 dev blocks, each followed by its rd3 and yd3 (R2 progress) evaluations.
Usage: make_r_configs.py drift | rdata | rdev
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_a6_configs import SEEDS, SHA, bc_adapter, frozen_online, step32_adapter  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "configs", "r")
ROUNDS = (64, 128, 256)
DRIFT = {"data_name": "a6", "select": {"yd3": 192, "yd5": 64}, "salt": "A6-R drift", "n_ref": 16, "cap": 256,
         "seed": 8, "batch_seqs": 512, "micro": 8, "out": "a6r"}


R3 = {"out": "r3", "data_name": "rs1", "panel": "rsc", "cap": 256, "batch_seqs": 512}
SCREEN = {"attempts": 16, "seed": 10, "chunk": 256, "target": 256, "max": 2048, "fallback": 128}
CACHE = {"attempts": 64, "batch": 8, "seed": 11}
BLOCK = {"prompts_per_round": 32, "rounds": 256, "cap": 256, "lr": 1e-5, "micro": 8, "batch_seqs": 512,
         "fresh": {"attempts": 64, "batch": 8}, "save_rounds": [0, 32, 64, 128, 192, 256], "mask_limit": 0.05}
CONF_SEEDS = (11, 22, 33)
DEV = {"light": (16, "runs/bc2rl-s0-bc-devlight/adapters/step0016.safetensors"),
       "extended": (128, "runs/bc2rl-s0-bc-devext/adapters/step0128.safetensors")}
ARMS = ("S", "I", "F", "O")


def write(name, cfg):
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=1, sort_keys=True)
        f.write("\n")
    return os.path.relpath(path, ROOT)


def branches(s):
    """(family tag, reference step, run_id) of every A5/A6 online branch of seed s."""
    out = []
    for init, step in (("light", 16), ("extended", 128)):
        for method in ("grpo", "filtered"):
            out.append((f"b{step}-{method}-M-s{s}", step, frozen_online(init, method, s)[0]["run_id"]))
        for arm in ("mask", "t3"):
            c = json.load(open(os.path.join(ROOT, "configs", "a6", f"on-{init}-grpo-{arm}-s{s}.json")))
            out.append((f"b{step}-grpo-{arm.upper()}-s{s}", step, c["run_id"]))
    for step in (32, 64):
        c = json.load(open(os.path.join(ROOT, "configs", "a6", f"on-step{step}-grpo-s{s}.json")))
        out.append((f"b{step}-grpo-M-s{s}", step, c["run_id"]))
    return out


def ref_item(s, step):
    a = step32_adapter(s) if step == 32 else bc_adapter(s, step)
    d = {"tag": f"b{step}-s{s}", "adapter": a}
    if a in SHA:
        d["sha256"] = SHA[a]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["drift", "rdata", "rdev"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.stage != "drift":
        return r3(a.stage)
    tasks = []
    for s in SEEDS:
        base, _ = frozen_online("light", "grpo", s)
        score = [{"ref": f"b{step}-s{s}", "tag": f"{fam}-r{r:03d}", "adapter": f"runs/{rid}/adapters/round{r:04d}.safetensors"}
                 for fam, step, rid in branches(s) for r in ROUNDS]
        c = {k: base[k] for k in ("prompt_variant", "model", "data")}
        c.update(run_id=f"bc2rl-r1-drift-s{s}", drift=dict(DRIFT, refs=[ref_item(s, st) for st in (16, 32, 64, 128)],
                                                           score=score))
        tasks.append({"id": f"r1-drift-s{s}", "steps": [{"name": "drift", "cmd": "drift",
                                                          "config": write(f"drift-s{s}.json", c)}]})
    with open(os.path.join(ROOT, "configs", "worklists", "r1-drift.json"), "w") as f:
        json.dump({"name": "r1-drift", "tasks": tasks}, f, indent=1)
        f.write("\n")
    print(f"r1-drift: {len(tasks)} tasks, {sum(len(json.load(open(os.path.join(ROOT, t['steps'][0]['config'])))['drift']['score']) for t in tasks)} scored checkpoints")


def r3(stage):
    base, _ = frozen_online("light", "grpo", 11)
    keys = lambda: {k: base[k] for k in ("prompt_variant", "model", "data")}
    if stage == "rdata":
        c = dict(keys(), run_id="bc2rl-r3-data", xdata={"name": "rs1", "seed": 20261030, "exclude_all_ext": True, "panels": {
            "rsc": {"n_inputs": 3, "count": 2048}, "rd3": {"n_inputs": 3, "count": 256},
            "rf3": {"n_inputs": 3, "count": 768}, "rf5": {"n_inputs": 5, "count": 128}}})
        print(write("xdata-rs1.json", c))
        return
    conf = []
    for s in CONF_SEEDS:
        for step in (16, 128):
            ad = bc_adapter(s, step)
            conf.append({"tag": f"b{step}-s{s}", "adapter": ad, "sha256": SHA[ad], "role": "conf"})
    dev = [{"tag": f"dev-b{st}-s7", "adapter": ad, "role": "dev"} for st, ad in DEV.values()]
    prep = dict(keys(), run_id="bc2rl-r3-prep", seed=7, rprep=dict(R3, screen=SCREEN, cache=CACHE,
                                                                   screen_checkpoints=conf, cache_checkpoints=conf + dev))
    tasks = [{"id": "r3-prep", "steps": [{"name": "rprep", "cmd": "rprep", "config": write("rprep.json", prep)}]}]
    for init, (step, ad) in DEV.items():
        rid = f"bc2rl-r3-dev-b{step}-s7"
        blk = dict(keys(), run_id=rid, seed=7, rsource=dict(
            BLOCK, data_name="rs1", panel="rsc", prep_out="r3", cache_tag=f"dev-b{step}-s7",
            init={"tag": f"dev-b{step}-s7", "adapter": ad}, stream_seed=7, sample_seed=7))
        fin = [{"tag": f"r3-dev-b{step}-s7-{arm}-r256", "adapter": f"runs/{rid}/adapters/{arm}/round0256.safetensors",
                "attempts": 32} for arm in ARMS]
        ev = lambda data, panels, cks, seed: dict(keys(), run_id=f"{rid}-ev-{panels[0]}", xeval={
            "data_name": data, "panels": panels, "checkpoints": cks, "seed": seed, "variant": "format", "cap": 256,
            "batch_seqs": 768, "emit_counts": False})
        rd = ev("rs1", ["rd3"], [{"tag": f"dev-b{step}-s7", "adapter": ad, "attempts": 48, "split": 24}] + fin, 6)
        prog = [{"tag": f"r2-dev-b{step}-s7", "adapter": ad, "attempts": 16}] + [
            {"tag": f"r2-dev-b{step}-s7-{arm}-r{r:03d}", "adapter": f"runs/{rid}/adapters/{arm}/round{r:04d}.safetensors",
             "attempts": 16} for arm in ARMS for r in BLOCK["save_rounds"][1:]]
        yd = ev("a6", ["yd3"], prog, 9)
        tasks.append({"id": f"r3-dev-b{step}-s7", "requires": ["rsource/r3/common_pool.json"], "steps": [
            {"name": "rsource", "cmd": "rsource", "config": write(f"rs-dev-{init}-s7.json", blk)},
            {"name": "xeval-rd3", "cmd": "xeval", "config": write(f"xeval-rd3-dev-{init}-s7.json", rd)},
            {"name": "xeval-yd3", "cmd": "xeval", "config": write(f"xeval-yd3-dev-{init}-s7.json", yd)}]})
    with open(os.path.join(ROOT, "configs", "worklists", "r3-dev.json"), "w") as f:
        json.dump({"name": "r3-dev", "tasks": tasks}, f, indent=1)
        f.write("\n")
    print(f"r3-dev: {len(tasks)} tasks")


if __name__ == "__main__":
    main()
