#!/usr/bin/env python3
"""Write the program-S configurations and worklists (protocol/AMENDMENT-A6-2026-10-26.md Section 13).

Stdlib only.
  prescreen (13.3)  configs/s3/xeval-prescreen-<init>-s11.json  worklist s2-prescreen
                    light / extended seed-11 BC adapters on ext data st1s (pre_pool, pre_dev, pre_conf), 16 attempts
  prescreen-extra (13.11.2)  configs/s3/xeval-prescreenx-<init>-s11.json  worklist s2-prescreen-x
                    the same adapters, attempts and seed on ext data st1x (pre_pool_x)
  dev (13.4)        configs/s3/on-<init>-<arm>-s11.json         run bc2rl-s3-<init>-<arm>-s11  worklist s3-dev
                    configs/s3/xeval-init-<init>-s11.json, xeval-<init>-<arm>-s11.json (dev panels sd3d, sr3d)
Every online config is the frozen M-arm GRPO config of its seed and initial adapter (the A5/A6 configuration) with only
the Section 13.4 keys changed: pool_ext, class_per_round, train_classes, class_grad_every, save_rounds.
Arms: A (pool A, 2N problems), B (pool B, 2N), AB (N + N; 16 + 16 per round), AB-mB (AB, B terms masked),
AB-mA (AB, A terms masked). Every arm has 2N unique problems and 8,192 / 2N passes per problem.
Usage: make_s3_configs.py prescreen
       make_s3_configs.py prescreen-extra
       make_s3_configs.py dev --pool N
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_a6_configs import INIT_STEP, SHA, bc_adapter, frozen_online  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "configs", "s3")
DEV_SEED = 11
ARMS = ("A", "B", "AB", "AB-mB", "AB-mA")
EVAL = {"variant": "format", "cap": 256, "seed": 6, "batch_seqs": 768, "emit_counts": False}
PRE_SEED = 7          # prescreen sampling seed: independent of every evaluation seed (6)
CLASS_GRAD_EVERY = 8


def write(name, cfg):
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=1, sort_keys=True)
        f.write("\n")
    return os.path.relpath(path, ROOT)


def worklist(name, tasks):
    with open(os.path.join(ROOT, "configs", "worklists", f"{name}.json"), "w") as f:
        json.dump({"name": name, "tasks": tasks}, f, indent=1)
        f.write("\n")
    print(f"{name}: {len(tasks)} tasks")


def xeval(base, run_id, data_name, panels, checkpoints, seed=EVAL["seed"]):
    c = {k: base[k] for k in ("prompt_variant", "model", "data")}
    c.update(run_id=run_id, xeval=dict(EVAL, data_name=data_name, panels=panels, checkpoints=checkpoints, seed=seed))
    return c


def arm_keys(arm, n):
    ab = {"name": "st1p", "classes": {"A": ["trA", n], "B": ["trB", n]}}
    keys = {"A": {"pool_ext": {"name": "st1p", "classes": {"A": ["trA", 2 * n]}}},
            "B": {"pool_ext": {"name": "st1p", "classes": {"B": ["trB", 2 * n]}}},
            "AB": {"pool_ext": ab, "class_per_round": {"A": 16, "B": 16}},
            "AB-mB": {"pool_ext": ab, "class_per_round": {"A": 16, "B": 16}, "train_classes": ["A"]},
            "AB-mA": {"pool_ext": ab, "class_per_round": {"A": 16, "B": 16}, "train_classes": ["B"]}}[arm]
    return dict(keys, class_grad_every=CLASS_GRAD_EVERY, save_rounds=[64, 128, 192, 256])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prescreen", "prescreen-extra", "dev"])
    ap.add_argument("--pool", type=int, help="N: problems per class in the AB arms (A / B arms use 2N)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    s = DEV_SEED
    base, _ = frozen_online("light", "grpo", s)
    if a.stage in ("prescreen", "prescreen-extra"):
        x = a.stage == "prescreen-extra"
        tasks = []
        for init in ("light", "extended"):
            ad = bc_adapter(s, INIT_STEP[init])
            ck = [{"tag": f"pre-b{INIT_STEP[init]}-s{s}", "adapter": ad, "attempts": 16, "sha256": SHA[ad]}]
            ev = xeval(base, f"bc2rl-s2-pre{'x' if x else ''}-{init}-s{s}", "st1x" if x else "st1s",
                       ["pre_pool_x"] if x else ["pre_pool", "pre_dev", "pre_conf"], ck, seed=PRE_SEED)
            tasks.append({"id": f"s2-pre{'x' if x else ''}-{init}-s{s}", "steps": [{"name": "xeval", "cmd": "xeval",
                          "config": write(f"xeval-prescreen{'x' if x else ''}-{init}-s{s}.json", ev)}]})
        worklist("s2-prescreen-x" if x else "s2-prescreen", tasks)
        return
    if not a.pool:
        sys.exit("dev needs --pool N (fixed in the S2 dated addition)")
    if (256 * 32) % (2 * a.pool):
        sys.exit(f"8,192 prompt slots are not a whole number of passes over 2N = {2 * a.pool}")
    tasks = []
    for init in ("light", "extended"):
        ad = bc_adapter(s, INIT_STEP[init])
        ck = [{"tag": f"b{INIT_STEP[init]}-s{s}", "adapter": ad, "attempts": 48, "split": 24, "sha256": SHA[ad]}]
        tasks.append({"id": f"s3-init-{init}-s{s}", "steps": [{"name": "xeval", "cmd": "xeval", "config": write(
            f"xeval-init-{init}-s{s}.json", xeval(base, f"bc2rl-s3-init-{init}-s{s}", "st1p", ["sd3d", "sr3d"], ck))}]})
        for arm in ARMS:
            rid = f"bc2rl-s3-{init}-{arm.lower()}-s{s}"
            c, src = frozen_online(init, "grpo", s)
            c["run_id"] = rid
            c["s3_source"] = src
            c["online"]["init"]["sha256"] = SHA[c["online"]["init"]["adapter"]]
            c["online"].update(arm_keys(arm, a.pool))
            fin = f"runs/{rid}/adapters/round0256.safetensors"
            ev = xeval(base, f"bc2rl-s3-ev-{init}-{arm.lower()}-s{s}", "st1p", ["sd3d", "sr3d"],
                       [{"tag": f"s3-b{INIT_STEP[init]}-{arm}-s{s}", "adapter": fin, "attempts": 32}])
            tasks.append({"id": f"s3-{init}-{arm.lower()}-s{s}", "steps": [
                {"name": "online", "cmd": "online", "config": write(f"on-{init}-{arm.lower()}-s{s}.json", c)},
                {"name": "xeval", "cmd": "xeval", "config": write(f"xeval-{init}-{arm.lower()}-s{s}.json", ev)}]})
    worklist("s3-dev", tasks)


if __name__ == "__main__":
    main()
