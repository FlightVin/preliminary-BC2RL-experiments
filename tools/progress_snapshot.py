#!/usr/bin/env python3
"""Deterministic progress snapshot (laptop-side orchestration tool; reads Slurm and local state only).

Writes slurm-agentic-engineering/reports/progress/<UTC stamp>.md with: BC2RL jobs in flight,
allocated GPU-hours from Slurm accounting (all requeue segments), guard accounting, checklist
counts and time left to the deadline. The orchestrator appends judgement (estimates) afterwards.
"""
import datetime as dt
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(os.path.dirname(HERE), "slurm-agentic-engineering")
DEADLINE = dt.datetime(2026, 10, 27, 9, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4)))  # EDT


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    return r.stdout + (r.stderr if r.returncode else "")


def ledger():
    rows = []
    for line in open(os.path.join(HARNESS, "state", "LEDGER.jsonl")):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def gpu_hours(jobids):
    if not jobids:
        return {}, 0.0
    out = sh(["ssh", "-n", "babel", f"sacct -D -X -n -P -j {','.join(jobids)} -o JobID,JobName,State,ElapsedRaw,AllocTRES"])
    per, tot = {}, 0.0
    for line in out.splitlines():
        f = line.split("|")
        if len(f) < 5 or not f[3].isdigit():
            continue
        m = re.search(r"gres/gpu=(\d+)", f[4])
        g = int(m.group(1)) if m else 0
        h = g * int(f[3]) / 3600
        per[f[1]] = per.get(f[1], 0.0) + h
        tot += h
    return per, tot


def main():
    now = dt.datetime.now(dt.timezone.utc)
    rows = [r for r in ledger() if r["expid"].startswith("bc2rl-")]
    per, tot = gpu_hours([r["jobid"] for r in rows])
    guard = sh([sys.executable, os.path.join(HERE, "bc2rl_guard.py"), "report"]).strip()
    q = sh(["ssh", "-n", "babel", "squeue -u $USER -h -n " + ",".join(f"ae-{r['expid']}" for r in rows[-60:]) +
            " -o '%i %j %T %M %L %R'"]) if rows else ""
    ck = open(os.path.join(HARNESS, "state", "CHECKLIST.md")).read()
    counts = {k: len(re.findall(rf"^\s*- \[{re.escape(k)}\]", ck, re.M)) for k in ("x", "~", "!", " ")}
    left = DEADLINE - now
    lines = [f"# BC2RL progress snapshot {now:%Y-%m-%d %H:%M} UTC",
             "", f"- Time to deadline (2026-10-27 09:00 EDT): {left.total_seconds() / 3600:.1f} h",
             f"- Guard: `{guard}`",
             f"- Checklist items: done {counts['x']}, active {counts['~']}, blocked {counts['!']}, pending {counts[' ']}",
             f"- BC2RL submissions: {len(rows)}; allocated GPU-hours (sacct, all segments, GPU jobs): {tot:.2f}",
             "", "## In flight (squeue)", "```", q.strip() or "(none)", "```",
             "", "## GPU-hours by experiment (sacct -D)", "| expid | GPU-h |", "|---|---:|"]
    lines += [f"| {k} | {v:.2f} |" for k, v in sorted(per.items()) if v > 0]
    os.makedirs(os.path.join(HARNESS, "reports", "progress"), exist_ok=True)
    path = os.path.join(HARNESS, "reports", "progress", f"{now:%Y%m%dT%H%M}Z.md")
    open(path, "w").write("\n".join(lines) + "\n")
    print(path)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
