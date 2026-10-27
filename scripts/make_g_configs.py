#!/usr/bin/env python3
"""Write the stage-G (GSPO) configurations and worklists (protocol/AMENDMENT-A6-2026-10-26.md Section 12).

Stdlib only. Every online config is the frozen M-arm GRPO config of its seed and initial adapter with only the keys
that Section 12 names changed:
  probe (12.4)  configs/g/probe-<init>-<objective>-s11.json   run bc2rl-gp-<init>-<objective>-s11   worklist g-probe
  dev   (12.5)  configs/g/on-<init>-<arm>-s11.json           run bc2rl-g-<init>-<arm>-s11            worklist g-dev
                configs/g/xeval-yd-s11.json (finals on yd3 / yd5)
Usage: make_g_configs.py probe --precision bf16|fp32
       make_g_configs.py dev --precision P --eps-tok X --eps-seq Y
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_a6_configs import EVAL, INIT_STEP, SHA, ck_final, frozen_online, xeval_cfg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "configs", "g")
DEV_SEED = 11
KL_EVERY = 8


def write(name, cfg):
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=1, sort_keys=True)
        f.write("\n")
    return os.path.relpath(path, ROOT)


def arm_keys(arm, eps_tok, eps_seq):
    return {"C1": {"objective": "grpo_const"},
            "L1": {"objective": "grpo_len", "clip_low": eps_tok, "clip_high": eps_tok},
            "L4": {"objective": "grpo_len", "updates_per_batch": 4, "clip_low": eps_tok, "clip_high": eps_tok},
            "S4": {"objective": "gspo", "updates_per_batch": 4, "clip_low": eps_seq, "clip_high": eps_seq}}[arm]


def online(init, seed, rid, keys, precision):
    c, src = frozen_online(init, "grpo", seed)
    c["run_id"] = rid
    c["g_source"] = src
    c["online"]["init"]["sha256"] = SHA[c["online"]["init"]["adapter"]]
    c["online"].update(keys, train_precision=precision, kl_every=KL_EVERY)
    return c


def worklist(name, tasks):
    with open(os.path.join(ROOT, "configs", "worklists", f"{name}.json"), "w") as f:
        json.dump({"name": name, "tasks": tasks}, f, indent=1)
        f.write("\n")
    print(f"{name}: {len(tasks)} tasks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["probe", "dev"])
    ap.add_argument("--precision", choices=["bf16", "fp32"], required=True)
    ap.add_argument("--eps-tok", type=float)
    ap.add_argument("--eps-seq", type=float)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    s = DEV_SEED
    if a.stage == "probe":
        tasks = []
        for init in ("light", "extended"):
            for obj in ("grpo_len", "gspo"):
                rid = f"bc2rl-gp-{init}-{obj.replace('_', '')}-s{s}"
                c = online(init, s, rid, {"objective": obj, "updates_per_batch": 4, "clip_low": None,
                                          "clip_high": None, "dump_ratios": True}, a.precision)
                c["online"].update(rounds=32, save_rounds=[32], val_rounds=[], endpoints={"online": 32})
                tasks.append({"id": f"gp-{init}-{obj}", "steps": [{"name": "online", "cmd": "online", "config":
                              write(f"probe-{init}-{obj.replace('_', '')}-s{s}.json", c)}]})
        worklist("g-probe", tasks)
        return
    if a.eps_tok is None or a.eps_seq is None:
        sys.exit("dev needs --eps-tok and --eps-seq (Section 12.4)")
    tasks, finals = [], []
    for init in ("light", "extended"):
        for arm in ("C1", "L1", "L4", "S4"):
            rid = f"bc2rl-g-{init}-{arm.lower()}-s{s}"
            c = online(init, s, rid, arm_keys(arm, a.eps_tok, a.eps_seq), a.precision)
            c["online"]["save_rounds"] = [64, 128, 256]
            tasks.append({"id": f"g-{init}-{arm.lower()}-s{s}", "steps": [{"name": "online", "cmd": "online",
                          "config": write(f"on-{init}-{arm.lower()}-s{s}.json", c)}]})
            finals.append(ck_final(f"g-b{INIT_STEP[init]}-{arm}-s{s}", f"runs/{rid}/adapters/round0256.safetensors"))
    base, _ = frozen_online("light", "grpo", s)
    ev = xeval_cfg(base, f"bc2rl-g-yd-s{s}", ["yd3", "yd5"], finals)
    assert ev["xeval"]["data_name"] == EVAL["data_name"] == "a6"
    tasks.append({"id": f"g-yd-s{s}", "requires": [f["adapter"] for f in finals],
                  "steps": [{"name": "xeval", "cmd": "xeval", "config": write(f"xeval-yd-s{s}.json", ev)}]})
    worklist("g-dev", tasks)


if __name__ == "__main__":
    main()
