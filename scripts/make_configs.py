#!/usr/bin/env python3
"""Write every run configuration from one place, so the frozen matrix is reproducible.

Usage: python scripts/make_configs.py <stage>   (stage0 | main)
"""
import os
import sys

import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = {"version": "v1", "main_seed": 20261024, "dev_seed": 1024, "min_value": 2, "max_value": 20,
        "max_target": 1000, "max_abs_intermediate": 10000, "ops": ["+", "-", "*"], "horizons": [3, 5, 10],
        "format_example_inputs": 4}
LORA = {"r": 16, "alpha": 32, "dropout": 0.0}
BASE = {"prompt_variant": "format", "model": {"lora": LORA}, "data": DATA}


def write(path, cfg):
    full = os.path.join(ROOT, "configs", path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:          # JSON is valid YAML; no PyYAML needed to write configs
        json.dump(cfg, f, indent=1)
        f.write("\n")
    print(path)


def ev(run_id, split, variant, cap, ckpts, attempts=15, seed=1):
    return dict(BASE, run_id=run_id, seed=0, evaluate={"split": split, "variant": variant, "attempts": attempts,
                                                       "cap": cap, "seed": seed, "batch_seqs": 480,
                                                       "checkpoints": ckpts})


def stage0():
    write("data_v1.yaml", dict(BASE, run_id="data-v1", seed=0))
    base = [{"tag": "base", "adapter": "base", "save_samples": True}]
    write("stage0/eval-base-format.yaml", ev("bc2rl-s0-eval-base-format", "dev", "format", 256, base))
    write("stage0/eval-base-zeroshot.yaml", ev("bc2rl-s0-eval-base-zeroshot", "dev", "zeroshot", 256, base))
    write("stage0/eval-base-format-cap512.yaml", ev("bc2rl-s0-eval-base-format-cap512", "dev", "format", 512, base))
    # disposable light-BC dev path: 16 steps on the 512 development demonstrations
    write("stage0/bc-devlight.yaml", dict(BASE, run_id="bc2rl-s0-bc-devlight", seed=7,
                                          bc={"split": "devdemo", "steps": 16, "batch": 32, "micro": 32, "lr": 1e-4,
                                              "save_steps": [16]},
                                          val={"split": "dev", "attempts": 4, "cap": 256}))
    # continuation of the same disposable dev BC path (seed 7, same order) to 128 steps: learnability gate
    write("stage0/bc-devext.yaml", dict(BASE, run_id="bc2rl-s0-bc-devext", seed=7,
                                        bc={"split": "devdemo", "steps": 128, "batch": 32, "micro": 8, "lr": 1e-4,
                                            "save_steps": [16, 32, 64, 128]},
                                        val={"split": "dev", "attempts": 4, "cap": 256}))
    write("stage0/eval-devext-format.yaml",
          ev("bc2rl-s0-eval-devext-format", "dev", "format", 256,
             [{"tag": f"devext{k}", "adapter": f"runs/bc2rl-s0-bc-devext/adapters/step{k:04d}.safetensors",
               "save_samples": k == 128} for k in (32, 64, 128)]))
    write("stage0/eval-devlight-format.yaml",
          ev("bc2rl-s0-eval-devlight-format", "dev", "format", 256,
             [{"tag": "devlight16", "adapter": "runs/bc2rl-s0-bc-devlight/adapters/step0016.safetensors",
               "save_samples": True}]))
    # LR calibration: same disposable init, equal rollout rounds, development prompts and selection set
    for method in ("grpo", "filtered"):
        for lr in (5e-6, 1e-5):
            rid = f"bc2rl-s0-cal-{method}-lr{lr:.0e}"
            write(f"stage0/cal-{method}-lr{lr:.0e}.yaml", dict(
                BASE, run_id=rid, seed=7,
                online={"method": method, "lr": lr, "rounds": 64, "prompts_per_round": 32, "group": 4,
                        "cap": 256, "micro": 8, "clip": 0.2, "prompt_split": "devdemo", "stream_seed": 7,
                        "init": {"adapter": "runs/bc2rl-s0-bc-devlight/adapters/step0016.safetensors",
                                 "tag": "devlight16"},
                        "save_rounds": [64], "val_rounds": [0, 32, 64]},
                val={"split": "dev", "attempts": 8, "cap": 256}))


def main_matrix():
    """Frozen main matrix. FROZEN values come from configs/frozen.json (written at the freeze)."""
    fz = json.load(open(os.path.join(ROOT, "configs", "frozen.json")))
    cap, R, total = fz["cap"], fz["rounds"], fz["total_budget"]
    online_frozen = fz["lr"] is not None and R is not None
    bc_steps = {"nobc": 0, "light": fz["light_steps"], "extended": fz["extended_steps"]}
    val = {"split": "val", "attempts": 4, "cap": cap}
    for seed in fz["seeds"]:
        write(f"main/bc-broad-s{seed}.yaml", dict(
            BASE, run_id=f"bc2rl-s1-bc-broad-s{seed}", seed=seed,
            bc={"split": "bc", "steps": fz["contbc_steps"], "batch": 32, "micro": 8, "lr": 1e-4,
                "save_steps": sorted({bc_steps["light"], 64, bc_steps["extended"], 192, fz["contbc_steps"]})},
            val=val))
        for init in ("nobc", "light", "extended") if online_frozen else ():
            if init == "nobc":
                ini = {"adapter": "none", "tag": "nobc"}
            else:
                ini = {"adapter": f"runs/bc2rl-s1-bc-broad-s{seed}/adapters/step{bc_steps[init]:04d}.safetensors",
                       "tag": f"{init}-s{seed}"}
            total_ep = total - bc_steps[init]
            for method in ("grpo", "filtered"):
                write(f"main/on-{init}-{method}-s{seed}.yaml", dict(
                    BASE, run_id=f"bc2rl-s2-{init}-{method}-s{seed}", seed=seed,
                    online={"method": method, "lr": fz["lr"][method], "rounds": R, "prompts_per_round": 32,
                            "group": 4, "cap": cap, "micro": 8, "clip": 0.2, "prompt_split": "online",
                            "stream_seed": seed, "init": ini,
                            "save_rounds": sorted({64, 128, total_ep, R}),
                            "val_rounds": sorted({0, 64, 128, total_ep, R}),
                            "endpoints": {"online": R, "total": total_ep}},
                    val=val))


if __name__ == "__main__":
    {"stage0": stage0, "main": main_matrix}[sys.argv[1]]()
