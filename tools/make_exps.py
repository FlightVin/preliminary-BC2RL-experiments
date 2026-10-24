#!/usr/bin/env python3
"""Write harness experiment files (slurm-agentic-engineering/exp/*.sh) for the frozen matrix.

Usage: make_exps.py stage2 [--time HH:MM:SS]
Each Stage 2 branch validates its own input (the BC run's DONE.json and the init adapter) before
training, because harness dependencies are `afterany`.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP = os.path.join(ROOT, "slurm-agentic-engineering", "exp")

HEAD = """#!/usr/bin/env bash
#EXP gpus=1 cpus=4 mem=32G time={time} part=preempt
# HYPOTHESIS: {hyp}
export PROJ_DATA="/data/user_data/$USER/BC2RL"
export PY="$PROJ_DATA/venv/bin/python" HF_HOME="$PROJ_DATA/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.
"""


def stage2(time="08:00:00"):
    fz = json.load(open(os.path.join(ROOT, "configs", "frozen.json")))
    out = []
    for seed in fz["seeds"]:
        for init in ("nobc", "light", "extended"):
            for method in ("grpo", "filtered"):
                cfg = f"configs/main/on-{init}-{method}-s{seed}.yaml"
                rid = f"bc2rl-s2-{init}-{method}-s{seed}"
                expid = f"bc2rl-s2-{init}{method}-main-s{seed}"
                c = json.load(open(os.path.join(ROOT, cfg)))
                pre = "true"
                if init != "nobc":
                    pre = (f"test -s $PROJ_DATA/runs/bc2rl-s1-bc-broad-s{seed}/DONE.json && "
                           f"test -s $PROJ_DATA/{c['online']['init']['adapter']}")
                body = HEAD.format(time=time, hyp=f"Stage 2 branch {init} x {method}, seed {seed}: "
                                   f"{c['online']['rounds']} rounds at lr {c['online']['lr']}; endpoints "
                                   f"{c['online']['endpoints']}.")
                body += f"""
phase precondition "true" '
    {pre} || {{ echo "PRECONDITION: init for {rid} missing (BC run not complete)"; exit 1; }}
'
phase online "test -s $PROJ_DATA/runs/{rid}/DONE.json && test -s $AE_RESULTS/summary.json" '
    nvidia-smi --query-gpu=name --format=csv,noheader
    $PY -m bc2rl online --config {cfg} --results-dir $AE_RESULTS
'
"""
                path = os.path.join(EXP, f"{expid}.sh")
                open(path, "w").write(body)
                out.append(expid)
    print("\n".join(out))


if __name__ == "__main__":
    args = sys.argv[1:]
    t = args[args.index("--time") + 1] if "--time" in args else "08:00:00"
    {"stage2": stage2}[args[0]](t)
