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


FREEZE = """
phase freeze "test -s $PROJ_DATA/frozen/protocol_frozen.json" '
    mkdir -p $PROJ_DATA/frozen
    $PY -c "import json, subprocess; f = json.load(open(\\"configs/frozen.json\\")); assert f[\\"lr\\"] and f[\\"rounds\\"]; f[\\"git_sha\\"] = subprocess.check_output([\\"git\\", \\"rev-parse\\", \\"HEAD\\"]).decode().strip(); json.dump(f, open(\\"$PROJ_DATA/frozen/protocol_frozen.json\\", \\"w\\"), indent=1)"
'
"""


def stage4(time="02:00:00"):
    """Locked test evaluation of the initial checkpoints (base, light, extended) and continued BC."""
    fz = json.load(open(os.path.join(ROOT, "configs", "frozen.json")))
    jobs = [("bc2rl-s4-evalbase-main-s0", "configs/main/eval-base.yaml", "bc2rl-s4-eval-base", "true")]
    for seed in fz["seeds"]:
        jobs.append((f"bc2rl-s4-evalbcpath-main-s{seed}", f"configs/main/eval-bcpath-s{seed}.yaml",
                     f"bc2rl-s4-eval-bcpath-s{seed}", f"test -s $PROJ_DATA/runs/bc2rl-s1-bc-broad-s{seed}/DONE.json"))
    for expid, cfg, eid, pre in jobs:
        body = HEAD.format(time=time, hyp=f"Stage 4 locked test evaluation ({eid}), 288 problems x 15 attempts.")
        body += f"""
phase precondition "true" '{pre} || {{ echo "PRECONDITION failed"; exit 1; }}'
""" + FREEZE + f"""
phase test-eval "test -s $PROJ_DATA/evals/{eid}/DONE.json" '
    $PY -m bc2rl evaluate --config {cfg}
'
"""
        open(os.path.join(EXP, f"{expid}.sh"), "w").write(body)
        print(expid)


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
""" + FREEZE + f"""
# locked test evaluation of this branch's frozen endpoints; outputs stay in $PROJ_DATA (not harvested)
phase test-eval "test -s $PROJ_DATA/evals/bc2rl-s4-eval-{init}-{method}-s{seed}/DONE.json" '
    $PY -m bc2rl evaluate --config configs/main/eval-{init}-{method}-s{seed}.yaml
'
"""
                path = os.path.join(EXP, f"{expid}.sh")
                open(path, "w").write(body)
                out.append(expid)
    print("\n".join(out))


if __name__ == "__main__":
    args = sys.argv[1:]
    t = args[args.index("--time") + 1] if "--time" in args else "08:00:00"
    {"stage2": stage2, "stage4": stage4}[args[0]](t)
