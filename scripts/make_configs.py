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
    return dict(BASE, run_id=run_id, seed=0, evaluate={"split": split, "allow_test": split == "test",
                                                       "variant": variant, "attempts": attempts,
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
            rid = f"bc2rl-s0-cal{3 if method == 'grpo' else 2}-{method}-lr{lr:.0e}"   # cal3: GRPO rerun after the same-pass ratio fix
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
    conds, train_runs = [], []
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
                rid = f"bc2rl-s2-{init}-{method}-s{seed}"
                cks = [{"tag": f"{init}-{method}-s{seed}-r{r}", "adapter": f"runs/{rid}/adapters/round{r:04d}.safetensors"}
                       for r in sorted({R, total_ep})]
                write(f"main/eval-{init}-{method}-s{seed}.yaml",
                      ev(f"bc2rl-s4-eval-{init}-{method}-s{seed}", "test", "format", cap, cks, seed=fz["eval_seed"]))
                for r in sorted({R, total_ep}):
                    eps = [e for e, v in (("online", R), ("total", total_ep)) if v == r]
                    for e in eps:
                        conds.append({"tag": f"{init}-{method}-s{seed}-r{r}", "eval_id": f"bc2rl-s4-eval-{init}-{method}-s{seed}",
                                      "init": init, "method": method, "seed": seed, "endpoint": e,
                                      "train_run": rid, "bc_steps": bc_steps[init], "online_rounds": r})
                train_runs.append({"run_id": rid, "kind": "online", "init": init, "method": method, "seed": seed})
        # initial checkpoints and the continued-BC endpoint of this seed's BC path
        bcr = f"bc2rl-s1-bc-broad-s{seed}"
        ick = [{"tag": f"light-s{seed}", "adapter": f"runs/{bcr}/adapters/step{bc_steps['light']:04d}.safetensors"},
               {"tag": f"extended-s{seed}", "adapter": f"runs/{bcr}/adapters/step{bc_steps['extended']:04d}.safetensors"},
               {"tag": f"contbc-s{seed}", "adapter": f"runs/{bcr}/adapters/step{fz['contbc_steps']:04d}.safetensors"}]
        write(f"main/eval-bcpath-s{seed}.yaml", ev(f"bc2rl-s4-eval-bcpath-s{seed}", "test", "format", cap, ick,
                                                    seed=fz["eval_seed"]))
        for t, init, m, ep, st in ((f"light-s{seed}", "light", "initial", "initial", bc_steps["light"]),
                                   (f"extended-s{seed}", "extended", "initial", "initial", bc_steps["extended"]),
                                   (f"contbc-s{seed}", "extended", "contbc", "bc256", fz["contbc_steps"])):
            conds.append({"tag": t, "eval_id": f"bc2rl-s4-eval-bcpath-s{seed}", "init": init, "method": m,
                          "seed": seed, "endpoint": ep, "train_run": bcr, "bc_steps": st, "online_rounds": 0})
        train_runs.append({"run_id": bcr, "kind": "bc", "init": "bc", "method": "bc", "seed": seed})
    if online_frozen:
        write("main/eval-base.yaml", ev("bc2rl-s4-eval-base", "test", "format", cap,
                                         [{"tag": "base", "adapter": "base"}], seed=fz["eval_seed"]))
        conds.insert(0, {"tag": "base", "eval_id": "bc2rl-s4-eval-base", "init": "nobc", "method": "initial",
                         "seed": None, "endpoint": "initial", "train_run": None, "bc_steps": 0, "online_rounds": 0})
        full = os.path.join(ROOT, "configs", "main", "matrix.json")
        json.dump({"conditions": conds, "train_runs": train_runs}, open(full, "w"), indent=1)
        print(f"main/matrix.json: {len(conds)} conditions, "
              f"{len({c['tag'] for c in conds})} unique evaluated checkpoints, {len(train_runs)} training runs")


def amend1():
    """Amendment A2 (2026-10-25): E3 = {light, extended} x {grpo, filtered} x 3 seeds under the frozen
    branch configs (unchanged), plus 15-attempt validation of each initial checkpoint and each round-256
    endpoint (retention and the E4 gate on validation). Writes the E3 worklist in priority order."""
    fz = json.load(open(os.path.join(ROOT, "configs", "frozen.json")))
    cap, R = fz["cap"], fz["rounds"]
    tasks = []
    for seed in fz["seeds"]:
        bcr = f"bc2rl-s1-bc-broad-s{seed}"
        init_ck = [{"tag": f"light-s{seed}", "adapter": f"runs/{bcr}/adapters/step{fz['light_steps']:04d}.safetensors"},
                   {"tag": f"extended-s{seed}", "adapter": f"runs/{bcr}/adapters/step{fz['extended_steps']:04d}.safetensors"}]
        write(f"amend1/val15-init-s{seed}.yaml", ev(f"bc2rl-e3-val15-init-s{seed}", "val", "format", cap, init_ck,
                                                    seed=fz["eval_seed"]))
        tasks.append({"id": f"init-s{seed}", "steps": [
            {"name": "test-eval-bcpath", "cmd": "evaluate", "results": False,
             "config": f"configs/main/eval-bcpath-s{seed}.yaml"},
            {"name": "val15-init", "cmd": "evaluate", "config": f"configs/amend1/val15-init-s{seed}.yaml"}]})
        for init, method in (("light", "grpo"), ("light", "filtered"), ("extended", "grpo"), ("extended", "filtered")):
            rid = f"bc2rl-s2-{init}-{method}-s{seed}"
            write(f"amend1/val15-{init}-{method}-s{seed}.yaml",
                  ev(f"bc2rl-e3-val15-{init}-{method}-s{seed}", "val", "format", cap,
                     [{"tag": f"{init}-{method}-s{seed}-r{R}", "adapter": f"runs/{rid}/adapters/round{R:04d}.safetensors"}],
                     seed=fz["eval_seed"]))
            steps = [{"name": "online", "cmd": "online", "config": f"configs/main/on-{init}-{method}-s{seed}.yaml"},
                     {"name": "test-eval", "cmd": "evaluate", "results": False,
                      "config": f"configs/main/eval-{init}-{method}-s{seed}.yaml"},
                     {"name": "val15", "cmd": "evaluate", "config": f"configs/amend1/val15-{init}-{method}-s{seed}.yaml"}]
            if (init, method, seed) == ("light", "grpo", 11):
                steps = steps[2:]            # trained and test-evaluated in job 10917856 (protocol 5deefef)
            tasks.append({"id": f"{init}-{method}-s{seed}", "steps": steps})
    full = os.path.join(ROOT, "configs", "worklists", "e3.json")
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        json.dump({"name": "e3", "tasks": tasks}, f, indent=1)
        f.write("\n")
    print(f"worklists/e3.json: {len(tasks)} tasks: {[t['id'] for t in tasks]}")


MECH = {"group": 4, "cap": 256, "micro": 8, "lr": 1e-05, "groups_per_arm": 8, "pool_prompts": 1024,
        "pool_seed": 20261025, "max_mixed": 512, "max_pools": 8, "pool_split_seed": 777, "len_tol": 0.05,
        "search_budget": 3000, "eval_seed": 1, "probe_attempts": 15, "kl_samples": 4, "batch_seqs": 480}


def mech():
    """E0/E2 configs (amendment A2 Section 4). rho comes from configs/mech/frozen_mech.json once calibrated."""
    write("mech/toy.yaml", dict(BASE, run_id="bc2rl-e0-toy", seed=0))
    write("mech/mechdata.yaml", dict(BASE, run_id="bc2rl-e2-mechdata", seed=0,
                                     mechdata={"seed": 20261025, "per_inputs": 16}))
    write("mech/calib.yaml", dict(BASE, run_id="bc2rl-e2-calib-devlight-s7", seed=7, mech=dict(
        MECH, mode="calibrate", adapter="runs/bc2rl-s0-bc-devlight/adapters/step0016.safetensors",
        source_splits=["dev", "devdemo"], pool_prompts=656, sample_seed=20261025 + 900, n_pairs=2,
        kl_split="dev", kl_target=0.01, rho_grid=[1, 2, 4, 8, 16, 32, 64, 128])))
    fzm = os.path.join(ROOT, "configs", "mech", "frozen_mech.json")
    if not os.path.exists(fzm):
        print("frozen_mech.json absent: E2 checkpoint configs not written (calibrate first)")
        return
    rho = json.load(open(fzm))["rho"]
    res = os.path.join(ROOT, "slurm-agentic-engineering", "results")
    for ci, seed in enumerate((11, 22, 33)):
        summ = json.load(open(os.path.join(res, f"bc2rl-s1-bcbroad-main-s{seed}", "summary.json")))
        sha = {c["step"]: c["sha256"] for c in summ["checkpoints"]}
        for name, step in (("light", 16), ("extended", 128)):
            write(f"mech/e2-{name}-s{seed}.yaml", dict(BASE, run_id=f"bc2rl-e2-{name}-s{seed}", seed=seed, mech=dict(
                MECH, mode="intervene", adapter=f"runs/bc2rl-s1-bc-broad-s{seed}/adapters/step{step:04d}.safetensors",
                adapter_sha256=sha[step], source_splits=["online"], sample_seed=20261025 + 10 * ci + (step == 128),
                n_pairs=4, rho=rho)))


if __name__ == "__main__":
    {"stage0": stage0, "main": main_matrix, "amend1": amend1, "mech": mech}[sys.argv[1]]()
