#!/usr/bin/env python3
"""BC2RL submission guard (laptop-side orchestration tool; never runs research code).

Enforces the PROJECT.md resource rules before every `ae submit`:
  * GPU jobs: gpus=1, part=preempt (preempt_qos), cpus<=4, mem<=32G, wall <= min(8h, site max).
  * BC2RL_cap = max(0, min(CEILING, L - max(9, A) - max(8, W) - U - BUFFER)), where
    L = verified preempt_qos MaxTRESPU gres/gpu, A = other ae study, W = wcw study,
    U = every other non-BC2RL preempt GPU commitment (unknown jobs count here).
  * Every nonterminal GPU request counts (requested GPUs for pending, allocated for active).
  * CPU-only work: one concurrent BC2RL job, <= 4 CPUs, inside verified shared headroom.
  * Ownership is decided by the harness ledger job IDs, never by a name prefix alone.

Usage:
  bc2rl_guard.py check  <expid> [--after a,b]   # exit 0 = allowed, 3 = refused, 4 = cannot verify
  bc2rl_guard.py submit <expid> [--after a,b]   # check under lock, `ae submit`, refresh Slurm
  bc2rl_guard.py report                         # print current accounting
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

CEILING = 5            # explicit BC2RL ceiling (PROJECT.md); 6 only after a documented release
OTHER_AE_RESERVED = 9  # the other ae study's entitlement
WCW_RESERVED = 8
BUFFER = 2
MAX_WALL_S = 8 * 3600
MAX_CPUS_GPU_JOB = 4
MAX_MEM_GPU_JOB_MIB = 32 * 1024
MAX_CPUS_CPU_JOB = 4
USER = "vineethb"
SELFTESTS = {"ae-smoke", "ae-smoke-dep"}
EXPID_RE = re.compile(r"^bc2rl-[a-z0-9]+-[a-z0-9.]+(?:-[a-z0-9.]+)*-s\d+$")
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
            "BOOT_FAIL", "DEADLINE", "PREEMPTED_TERMINAL"}

HARNESS = os.environ.get("AE_HOME") or os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "slurm-agentic-engineering")


class Unverifiable(Exception):
    pass


@dataclass
class Job:
    jobid: str
    name: str
    state: str
    partition: str
    qos: str
    gres: str
    cpus: int
    reason: str = ""

    @property
    def gpus(self) -> int:
        return gpus_from_gres(self.gres)


@dataclass
class Accounting:
    L: int
    cpu_user_limit: int
    ours_gpu: int = 0
    A: int = 0
    W: int = 0
    U: int = 0
    ours_cpu_jobs: list = field(default_factory=list)
    others_cpu: int = 0
    ours: list = field(default_factory=list)
    unknown: list = field(default_factory=list)

    @property
    def cap(self) -> int:
        return max(0, min(CEILING, self.L - max(OTHER_AE_RESERVED, self.A)
                          - max(WCW_RESERVED, self.W) - self.U - BUFFER))


def gpus_from_gres(gres: str) -> int:
    """'gres/gpu:1', 'gres/gpu:L40S:2', 'gpu:1', 'N/A', '' -> GPU count. Unknown forms raise."""
    g = gres.strip()
    if g in ("", "N/A", "(null)"):
        return 0
    total = 0
    for part in g.split(","):
        part = part.strip()
        if "gpu" not in part:
            continue
        m = re.search(r"gpu(?::[A-Za-z0-9_]+)?[:=](\d+)", part)
        if not m:
            raise Unverifiable(f"cannot parse GPU request '{gres}'")
        total += int(m.group(1))
    return total


def parse_squeue(text: str) -> list[Job]:
    jobs = []
    for line in text.splitlines():
        if not line.strip():
            continue
        f = line.split("|")
        if len(f) < 7:
            raise Unverifiable(f"bad squeue line: {line!r}")
        jobs.append(Job(f[0], f[1], f[2], f[3], f[4], f[5], int(f[6] or 0), f[7] if len(f) > 7 else ""))
    return jobs


def parse_limits(text: str) -> tuple[int, int]:
    """sacctmgr -n -P show qos preempt_qos,preempt_cpu_qos format=Name,MaxTRESPU."""
    L = cpu = None
    for line in text.splitlines():
        name, _, tres = line.partition("|")
        if name == "preempt_qos":
            m = re.search(r"gres/gpu=(\d+)", tres)
            L = int(m.group(1)) if m else None
        elif name == "preempt_cpu_qos":
            m = re.search(r"cpu=(\d+)", tres)
            cpu = int(m.group(1)) if m else None
    if L is None or cpu is None:
        raise Unverifiable(f"cannot verify QOS limits from: {text!r}")
    return L, cpu


def ledger_jobids(path: str) -> dict[str, str]:
    """jobid -> expid for every BC2RL (and selftest) submission in the harness ledger."""
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        e, j = row.get("expid", ""), str(row.get("jobid", ""))
        if j and (e.startswith("bc2rl-") or e in SELFTESTS):
            out[j] = e
    return out


def account(jobs: list[Job], L: int, cpu_limit: int, ours: dict[str, str],
            reserved: list[dict]) -> Accounting:
    acc = Accounting(L=L, cpu_user_limit=cpu_limit)
    for j in jobs:
        if j.state in TERMINAL:
            continue
        mine = j.jobid in ours
        on_preempt_gpu = j.partition == "preempt" and j.qos == "preempt_qos"
        if j.qos == "preempt_cpu_qos":
            if mine:
                acc.ours_cpu_jobs.append(j)
            else:
                acc.others_cpu += j.cpus
            continue
        if not on_preempt_gpu:
            continue                      # general/normal etc.: not part of L
        if mine:
            acc.ours_gpu += j.gpus
            acc.ours.append(j)
        elif j.name.startswith("wcw-"):
            acc.W += j.gpus
        elif j.name.startswith("ae-") and not j.name.startswith("ae-bc2rl-"):
            acc.A += j.gpus
        else:
            acc.U += j.gpus               # unknown -> external capacity
            acc.unknown.append(j)
    seen = {j.jobid for j in jobs}
    for r in reserved:                    # local reservations not yet visible in squeue
        if r.get("jobid") and r["jobid"] in seen:
            continue
        acc.ours_gpu += int(r.get("gpus", 0))
    return acc


def parse_header(expfile: str) -> dict[str, str]:
    meta = {}
    for line in open(expfile):
        if line.startswith("#EXP "):
            for tok in line[5:].split():
                k, _, v = tok.partition("=")
                meta[k] = v
    return meta


def hms(s: str) -> int:
    d = 0
    if "-" in s:
        dd, s = s.split("-", 1)
        d = int(dd)
    parts = [int(x) for x in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return d * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def mem_mib(s: str) -> int:
    m = re.fullmatch(r"(\d+)([KMGT]?)B?", s.upper())
    if not m:
        raise ValueError(f"bad mem '{s}'")
    n, u = int(m.group(1)), m.group(2) or "M"
    return {"K": n // 1024, "M": n, "G": n * 1024, "T": n * 1024 * 1024}[u]


def check_recipe(expid: str, meta: dict[str, str]) -> list[str]:
    errs = []
    if expid not in SELFTESTS and not EXPID_RE.match(expid):
        errs.append(f"expid '{expid}' does not match bc2rl-<phase>-<condition>-s<seed>")
    if "array" in meta:
        errs.append("job arrays are prohibited")
    gpus = int(meta.get("gpus", "1"))
    cpus = int(meta.get("cpus", "4"))
    if "time" not in meta:
        errs.append("recipe must declare time=")
    elif hms(meta["time"]) > MAX_WALL_S:
        errs.append(f"time={meta['time']} exceeds 8 h")
    if gpus == 0:
        if cpus > MAX_CPUS_CPU_JOB:
            errs.append(f"CPU job requests {cpus} > {MAX_CPUS_CPU_JOB} CPUs")
        return errs
    if gpus != 1:
        errs.append(f"gpus={gpus}: one GPU per job only")
    if meta.get("part") != "preempt":
        errs.append("GPU recipe must declare part=preempt explicitly")
    if cpus > MAX_CPUS_GPU_JOB:
        errs.append(f"cpus={cpus} > {MAX_CPUS_GPU_JOB}")
    if "mem" not in meta:
        errs.append("GPU recipe must declare mem=")
    elif mem_mib(meta["mem"]) > MAX_MEM_GPU_JOB_MIB:
        errs.append(f"mem={meta['mem']} > 32G")
    if "gpus=" in meta.get("gtype", ""):
        errs.append("bad gtype")
    return errs


def decide(expid: str, meta: dict[str, str], acc: Accounting, after: list[str],
           ours: dict[str, str]) -> list[str]:
    errs = check_recipe(expid, meta)
    gpus = int(meta.get("gpus", "1"))
    if gpus > 0:
        if acc.ours_gpu > acc.cap:
            errs.append(f"OVER CAP: BC2RL holds {acc.ours_gpu} > cap {acc.cap}; release before submitting")
        elif acc.ours_gpu + gpus > acc.cap:
            errs.append(f"cap: BC2RL {acc.ours_gpu} + {gpus} > cap {acc.cap} "
                        f"(L={acc.L} A={acc.A} W={acc.W} U={acc.U})")
    else:
        own_live = {j.jobid for j in acc.ours_cpu_jobs}
        chained = any(ours_job_for(a, ours) in own_live for a in after)
        if acc.ours_cpu_jobs and not chained:
            errs.append(f"CPU: BC2RL already has {len(acc.ours_cpu_jobs)} nonterminal CPU job(s)")
        if acc.cpu_user_limit - acc.others_cpu - sum(j.cpus for j in acc.ours_cpu_jobs) < int(meta.get("cpus", "4")):
            errs.append("CPU: no verified shared CPU headroom")
    return errs


def ours_job_for(expid: str, ours: dict[str, str]) -> str | None:
    hits = [j for j, e in ours.items() if e == expid]
    return max(hits, key=int) if hits else None


# ---------------------------------------------------------------- live I/O

def sh(cmd: list[str], timeout=120) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise Unverifiable(f"{' '.join(cmd)} failed: {r.stderr.strip()}")
    return r.stdout


def live_state():
    q = sh(["ssh", "-n", "-o", "BatchMode=yes", "babel",
            f"squeue -u {USER} -h -o '%i|%j|%T|%P|%q|%b|%C|%r' && echo __SPLIT__ && "
            "sacctmgr -n -P show qos preempt_qos,preempt_cpu_qos format=Name,MaxTRESPU"])
    sq, _, lim = q.partition("__SPLIT__\n")
    jobs = parse_squeue(sq)
    L, cpu = parse_limits(lim)
    return jobs, L, cpu


RES_FILE = os.path.join(HARNESS, "state", "bc2rl-reservations.json")
LOCK_FILE = os.path.join(HARNESS, "state", ".bc2rl-submit.lock")


def load_res():
    try:
        return json.load(open(RES_FILE))
    except (OSError, json.JSONDecodeError):
        return []


def save_res(rows):
    tmp = RES_FILE + ".tmp"
    json.dump(rows, open(tmp, "w"), indent=1)
    os.replace(tmp, RES_FILE)


def summary(acc: Accounting) -> str:
    return (f"L={acc.L} A={acc.A} W={acc.W} U={acc.U} cap={acc.cap} BC2RL_gpu={acc.ours_gpu} "
            f"BC2RL_cpu_jobs={len(acc.ours_cpu_jobs)} others_cpu={acc.others_cpu}/{acc.cpu_user_limit}"
            + (f" unknown={[j.jobid + ':' + j.name for j in acc.unknown]}" if acc.unknown else ""))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "submit", "report"])
    ap.add_argument("expid", nargs="?")
    ap.add_argument("--after", default="")
    a = ap.parse_args(argv)
    ledger = os.path.join(HARNESS, "state", "LEDGER.jsonl")
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    with open(LOCK_FILE, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            jobs, L, cpu = live_state()
        except Unverifiable as e:
            print(f"GUARD: cannot verify accounting: {e}", file=sys.stderr)
            return 4
        ours = ledger_jobids(ledger)
        res = [r for r in load_res() if time.time() - r.get("t", 0) < 600]
        acc = account(jobs, L, cpu, ours, res)
        print("GUARD:", summary(acc))
        if a.cmd == "report":
            return 0 if acc.ours_gpu <= acc.cap else 5
        after = [x for x in a.after.split(",") if x]
        expfile = os.path.join(HARNESS, "exp", f"{a.expid}.sh")
        if not os.path.exists(expfile):
            expfile = os.path.join(HARNESS, "selftest", f"{a.expid}.sh")
        meta = parse_header(expfile)
        errs = decide(a.expid, meta, acc, after, ours)
        if errs:
            for e in errs:
                print("GUARD REFUSED:", e, file=sys.stderr)
            return 3
        print(f"GUARD: allowed {a.expid} ({meta})")
        if a.cmd == "check":
            return 0
        gpus = int(meta.get("gpus", "1"))
        res.append({"expid": a.expid, "gpus": gpus, "t": time.time(), "jobid": None})
        save_res(res)
        cmd = [os.path.join(HARNESS, "bin", "ae"), "submit", a.expid] + (["--after", a.after] if a.after else [])
        r = subprocess.run(cmd)
        ours = ledger_jobids(ledger)
        jid = ours_job_for(a.expid, ours)
        res[-1]["jobid"] = jid
        if r.returncode != 0:
            res.pop()
        save_res(res)
        jobs, L, cpu = live_state()
        acc = account(jobs, L, cpu, ours, res)
        print("GUARD after submit:", summary(acc), f"job={jid}")
        if acc.ours_gpu > acc.cap:
            print("GUARD WARNING: BC2RL above cap after submission", file=sys.stderr)
            return 5
        return r.returncode


if __name__ == "__main__":
    sys.exit(main())
