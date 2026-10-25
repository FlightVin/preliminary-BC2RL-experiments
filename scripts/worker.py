#!/usr/bin/env python3
"""Bounded sequential worker: run several registered tasks inside one Slurm allocation.

Usage: worker.py --worklist configs/worklists/<name>.json --results-dir $AE_RESULTS

Each task keeps its own configuration, run_id, checkpoints, logs and cost ledger (the `bc2rl`
commands own those). This script only orders the tasks and prevents duplicate execution:

* A task is claimed with an atomic `mkdir $PROJ_DATA/claims/<worklist>/<task>`; the claim holds the
  owning SLURM_JOB_ID. A task claimed by another job is skipped (never taken over automatically).
  A requeued job keeps its job ID, so it resumes its own claimed task first.
* Each step is a `bc2rl` command whose run/eval directory has its own done-predicate, so a resumed
  task skips finished steps in seconds.
* A signal exit (status > 128, e.g. preemption) stops the worker with that status so Slurm requeues
  it. A deterministic failure marks the task FAILED, and the worker continues with the next task;
  the job then exits 1 so the orchestrator sees the failure.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worklist", required=True)
    ap.add_argument("--results-dir", required=True)
    a = ap.parse_args()
    wl = json.load(open(a.worklist))
    pd = os.environ["PROJ_DATA"]
    py = os.environ.get("PY", sys.executable)
    job = os.environ.get("SLURM_JOB_ID", "local")
    croot = os.path.join(pd, "claims", wl["name"])
    os.makedirs(croot, exist_ok=True)
    os.makedirs(a.results_dir, exist_ok=True)
    # The child receives Slurm's SIGTERM directly; the worker only waits for it to checkpoint.
    signal.signal(signal.SIGTERM, lambda s, f: None)
    signal.signal(signal.SIGUSR1, lambda s, f: None)
    status = {"worklist": wl["name"], "job": job, "tasks": {}}
    failed = False
    for t in wl["tasks"]:
        tid = t["id"]
        cdir = os.path.join(croot, tid)
        if os.path.exists(os.path.join(cdir, "DONE")):
            status["tasks"][tid] = "done-before"
            continue
        try:
            os.mkdir(cdir)
            open(os.path.join(cdir, "owner"), "w").write(job)
        except FileExistsError:
            owner = open(os.path.join(cdir, "owner")).read().strip() if os.path.exists(
                os.path.join(cdir, "owner")) else "?"
            if owner != job:
                print(f"[worker] {tid}: claimed by job {owner}; skip", flush=True)
                status["tasks"][tid] = f"claimed-by-{owner}"
                continue
            print(f"[worker] {tid}: resuming own claim", flush=True)
        if os.path.exists(os.path.join(cdir, "FAILED")):
            print(f"[worker] {tid}: marked FAILED earlier; skip (orchestrator decides)", flush=True)
            status["tasks"][tid] = "failed-before"
            continue
        tres = os.path.join(a.results_dir, tid)
        os.makedirs(tres, exist_ok=True)
        for k, step in enumerate(t["steps"]):
            cmd = [py, "-m", "bc2rl", step["cmd"], "--config", step["config"]]
            if step.get("results", True):
                cmd += ["--results-dir", os.path.join(tres, step.get("name", f"step{k}"))]
            print(f"[worker] {now()} {tid} step {k} ({step.get('name', '')}): {' '.join(cmd)}", flush=True)
            t0 = time.time()
            p = subprocess.Popen(cmd)
            while True:
                try:
                    rc = p.wait()
                    break
                except InterruptedError:
                    continue
            print(f"[worker] {now()} {tid} step {k} exit {rc} after {time.time() - t0:.0f}s", flush=True)
            if rc != 0:
                if rc < 0 or rc > 128:
                    json.dump(status, open(os.path.join(a.results_dir, "worker_status.json"), "w"), indent=1)
                    print(f"[worker] signal exit {rc}: stopping for requeue", flush=True)
                    sys.exit(143)
                open(os.path.join(cdir, "FAILED"), "w").write(f"{job} step {k} rc {rc} {now()}\n")
                status["tasks"][tid] = f"FAILED step {k} rc {rc}"
                failed = True
                break
        else:
            open(os.path.join(cdir, "DONE"), "w").write(f"{job} {now()}\n")
            status["tasks"][tid] = "done"
        json.dump(status, open(os.path.join(a.results_dir, "worker_status.json"), "w"), indent=1)
    json.dump(status, open(os.path.join(a.results_dir, "worker_status.json"), "w"), indent=1)
    print(f"[worker] finished: {json.dumps(status['tasks'])}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
