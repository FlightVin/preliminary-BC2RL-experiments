#!/usr/bin/env python3
"""Write the amendment-A6 configurations and the worker worklist (protocol/AMENDMENT-A6-2026-10-26.md).

Stdlib only; it derives every config from a frozen one and changes only the keys that the protocol names:
  BC path re-runs (Section 3.2)      configs/a6/bc-path-s<seed>.json        run bc2rl-y4-bc-path-s<seed>
  Y3 arms MASK / T3 (Section 6.1)    configs/a6/on-<init>-grpo-<arm>-s<seed>.json
  Y4 steps 32 / 64 (Section 8.1)     configs/a6/on-step<N>-grpo-s<seed>.json
  evaluations (Section 3.3)          configs/a6/xeval-*.json (data a6, cap 256 + continuation to 1024)
  worklist                           configs/worklists/a6.json
Adapter hashes: configs/a6/adapter_sha256.json (harvested BC / online summaries).
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "configs", "a6")
SEEDS = (11, 22, 33, 44, 55, 66)
INIT_STEP = {"light": 16, "extended": 128}
SHA = json.load(open(os.path.join(OUT, "adapter_sha256.json")))
EVAL = {"data_name": "a6", "variant": "format", "cap": 256, "cap_ext": 1024, "seed": 6, "batch_seqs": 768,
        "ext_batch_seqs": 256, "emit_counts": False}
T3E_ROUND = 85                   # 85 x 32 = 2,720 3-input prompts; the mixed stream holds 2 x 1,366 = 2,732


def frozen_bc(seed):
    p = f"configs/main/bc-broad-s{seed}.yaml" if seed <= 33 else f"configs/ext/x7/bc-broad-s{seed}.json"
    return json.load(open(os.path.join(ROOT, p))), p


def frozen_online(init, method, seed):
    p = (f"configs/main/on-{init}-{method}-s{seed}.yaml" if seed <= 33
         else f"configs/ext/x7/on-{init}-{method}-s{seed}.json")
    return json.load(open(os.path.join(ROOT, p))), p


def bc_adapter(seed, step):
    return f"runs/bc2rl-s1-bc-broad-s{seed}/adapters/step{step:04d}.safetensors"


def step32_adapter(seed):
    return f"runs/bc2rl-y4-bc-path-s{seed}/adapters/step0032.safetensors"


def write(name, cfg):
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=1, sort_keys=True)
        f.write("\n")
    return os.path.relpath(path, ROOT)


def xeval_cfg(base, run_id, panels, checkpoints):
    c = {k: base[k] for k in ("prompt_variant", "model", "data")}
    c.update(run_id=run_id, xeval=dict(EVAL, panels=panels, checkpoints=checkpoints))
    return c


def ck_init(seed, step, adapter=None):
    a = adapter or bc_adapter(seed, step)
    d = {"tag": f"b{step}-s{seed}", "adapter": a, "attempts": 48, "split": 24}
    if a in SHA:
        d["sha256"] = SHA[a]
    return d


def ck_final(tag, adapter):
    d = {"tag": tag, "adapter": adapter, "attempts": 32}
    if adapter in SHA:
        d["sha256"] = SHA[adapter]
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    tasks = {"bc": [], "yd": [], "y3": [], "y4_64": [], "init32": [], "y4_32": [], "yf": []}
    base_bc, _ = frozen_bc(11)
    # BC path re-runs (Section 3.2): frozen config, 128 steps, checkpoints 16/32/64/128, new run id
    bc_steps = []
    for s in SEEDS:
        c, src = frozen_bc(s)
        c["run_id"] = f"bc2rl-y4-bc-path-s{s}"
        c["bc"]["steps"] = 128
        c["bc"]["save_steps"] = [16, 32, 64, 128]
        c["a6_source"] = src
        bc_steps.append({"name": f"bc-s{s}", "cmd": "bc", "config": write(f"bc-path-s{s}.json", c)})
    tasks["bc"].append({"id": "bc-path", "steps": bc_steps})
    for s in SEEDS:
        finals_m = [ck_final(f"b{INIT_STEP[i]}-{m}-M-s{s}", f"runs/bc2rl-s2-{i}-{m}-s{s}/adapters/round0256.safetensors")
                    for i in ("light", "extended") for m in ("grpo", "filtered")]
        inits = [ck_init(s, n) for n in (16, 64, 128)]
        # development panels (Sections 5.1, 8.2)
        tasks["yd"].append({"id": f"yd-init-s{s}", "steps": [{"name": "xeval", "cmd": "xeval", "config": write(
            f"xeval-yd-init-s{s}.json", xeval_cfg(base_bc, f"bc2rl-a6-yd-init-s{s}", ["yd3", "yd5"], inits))}]})
        tasks["yd"].append({"id": f"yd-fin-s{s}", "steps": [{"name": "xeval", "cmd": "xeval", "config": write(
            f"xeval-yd-fin-s{s}.json", xeval_cfg(base_bc, f"bc2rl-a6-yd-fin-s{s}", ["yd3"], finals_m))}]})
        # step-32 adapter: development and confirmation panels (needs the BC re-run)
        c32 = [ck_init(s, 32, step32_adapter(s))]
        tasks["init32"].append({"id": f"init32-s{s}", "requires": [step32_adapter(s)], "steps": [
            {"name": "xeval-yd", "cmd": "xeval", "config": write(
                f"xeval-yd-init32-s{s}.json", xeval_cfg(base_bc, f"bc2rl-a6-yd-init32-s{s}", ["yd3", "yd5"], c32))},
            {"name": "xeval-yf", "cmd": "xeval", "config": write(
                f"xeval-yf-init32-s{s}.json", xeval_cfg(base_bc, f"bc2rl-a6-yf-init32-s{s}", ["yf3", "yf5"], c32))}]})
        # confirmation panels for the existing adapters
        tasks["yf"].append({"id": f"yf-init-s{s}", "steps": [{"name": "xeval", "cmd": "xeval", "config": write(
            f"xeval-yf-init-s{s}.json", xeval_cfg(base_bc, f"bc2rl-a6-yf-init-s{s}", ["yf3", "yf5"], inits))}]})
        tasks["yf"].append({"id": f"yf-fin-s{s}", "steps": [{"name": "xeval", "cmd": "xeval", "config": write(
            f"xeval-yf-fin-s{s}.json", xeval_cfg(base_bc, f"bc2rl-a6-yf-fin-s{s}", ["yf3", "yf5"], finals_m))}]})
        # Y3 arms (Section 6.1): MASK and T3 from steps 16 and 128, GRPO
        for i in ("light", "extended"):
            for arm in ("MASK", "T3"):
                c, src = frozen_online(i, "grpo", s)
                rid = f"bc2rl-y3-{i}-grpo-{arm.lower()}-s{s}"
                c["run_id"] = rid
                c["a6_source"] = src
                c["online"]["init"]["sha256"] = SHA[c["online"]["init"]["adapter"]]
                if arm == "MASK":
                    c["online"]["train_inputs"] = [3]
                else:
                    c["online"]["pool_inputs"] = [3]
                    c["online"]["save_rounds"] = sorted(set(c["online"]["save_rounds"]) | {T3E_ROUND})
                cks = [ck_final(f"b{INIT_STEP[i]}-grpo-{arm}-s{s}", f"runs/{rid}/adapters/round0256.safetensors")]
                if arm == "T3":
                    cks.append(ck_final(f"b{INIT_STEP[i]}-grpo-T3E-s{s}",
                                        f"runs/{rid}/adapters/round{T3E_ROUND:04d}.safetensors"))
                tasks["y3"].append({"id": f"y3-{i}-{arm.lower()}-s{s}", "steps": [
                    {"name": "online", "cmd": "online", "config": write(f"on-{i}-grpo-{arm.lower()}-s{s}.json", c)},
                    {"name": "xeval", "cmd": "xeval", "config": write(
                        f"xeval-y3-{i}-{arm.lower()}-s{s}.json",
                        xeval_cfg(base_bc, f"bc2rl-a6-yf-y3-{i}-{arm.lower()}-s{s}", ["yf3", "yf5"], cks))}]})
        # Y4 (Section 8.1): GRPO from steps 32 and 64
        for n in (64, 32):
            c, src = frozen_online("light", "grpo", s)
            rid = f"bc2rl-y4-step{n}-grpo-s{s}"
            ad = step32_adapter(s) if n == 32 else bc_adapter(s, 64)
            c["run_id"] = rid
            c["a6_source"] = src
            c["online"]["init"] = {"adapter": ad, "tag": f"step{n}-s{s}"}
            if ad in SHA:
                c["online"]["init"]["sha256"] = SHA[ad]
            c["online"]["save_rounds"] = [64, 128, 256]
            c["online"]["val_rounds"] = [0, 64, 128, 256]
            c["online"]["endpoints"] = {"online": 256}
            t = {"id": f"y4-step{n}-s{s}", "steps": [
                {"name": "online", "cmd": "online", "config": write(f"on-step{n}-grpo-s{s}.json", c)},
                {"name": "xeval", "cmd": "xeval", "config": write(
                    f"xeval-y4-step{n}-s{s}.json",
                    xeval_cfg(base_bc, f"bc2rl-a6-yf-y4-step{n}-s{s}", ["yf3", "yf5"],
                              [ck_final(f"b{n}-grpo-M-s{s}", f"runs/{rid}/adapters/round0256.safetensors")]))}]}
            if n == 32:
                t["requires"] = [ad]
                tasks["y4_32"].append(t)
            else:
                tasks["y4_64"].append(t)
    order = tasks["bc"] + tasks["yd"] + tasks["init32"] + tasks["y3"] + tasks["y4_64"] + tasks["y4_32"] + tasks["yf"]
    wl = {"name": "a6", "tasks": order}
    with open(os.path.join(ROOT, "configs", "worklists", "a6.json"), "w") as f:
        json.dump(wl, f, indent=1)
        f.write("\n")
    # data config (Section 3.1)
    d = {k: base_bc[k] for k in ("prompt_variant", "model", "data")}
    d.update(run_id="bc2rl-a6-data", xdata={"name": "a6", "seed": 20261027, "exclude_ext": ["x5"], "panels": {
        "yd3": {"n_inputs": 3, "count": 384}, "yd5": {"n_inputs": 5, "count": 96},
        "yf3": {"n_inputs": 3, "count": 768}, "yf5": {"n_inputs": 5, "count": 128}}})
    write("ydata.json", d)
    print(f"{len(order)} tasks: " + ", ".join(f"{k} {len(v)}" for k, v in tasks.items()))


if __name__ == "__main__":
    main()
