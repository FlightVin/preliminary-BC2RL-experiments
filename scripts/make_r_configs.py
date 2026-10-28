#!/usr/bin/env python3
"""Write the stage-R configurations and worklists (protocol/AMENDMENT-A6-2026-10-26.md Section 14). Stdlib only.

  drift (14.1)  configs/r/drift-s<seed>.json   run bc2rl-r1-drift-s<seed>   worklist r1-drift (one task per seed)
                references: the BC adapters of steps 16, 32 (re-run), 64, 128 of the seed;
                scored: rounds 64, 128, 256 of every A5/A6 branch of the seed (M GRPO and filtered, Y3 MASK and T3, Y4 steps
                32 and 64), each against its own BC reference. Round 0 is the reference itself (KL 0 by definition).
Usage: make_r_configs.py drift
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
    ap.add_argument("stage", choices=["drift"])
    ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
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


if __name__ == "__main__":
    main()
