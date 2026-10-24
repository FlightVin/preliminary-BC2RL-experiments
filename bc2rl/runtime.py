"""Run directories, configuration hashing, atomic checkpoints, cost ledger and signal handling."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import random
import signal
import socket
import time

import numpy as np
import torch
import yaml

CKPT_INTERVAL_S = 300          # full-state checkpoint at most every 5 min (PROJECT.md limit: 10 min)


def load_config(path):
    text = open(path).read()
    try:
        cfg = json.loads(text)        # configs are written as JSON; PyYAML (YAML 1.1) reads 5e-06 as a string
    except json.JSONDecodeError:
        cfg = yaml.safe_load(text)
    cfg.setdefault("proj_data", os.environ.get("PROJ_DATA", ""))
    if not cfg["proj_data"]:
        raise SystemExit("proj_data unset (config or $PROJ_DATA)")
    return cfg


def config_hash(cfg):
    c = {k: v for k, v in copy.deepcopy(cfg).items() if not k.startswith("_")}
    c.pop("proj_data", None)                 # location, not science; "_" keys are run plumbing
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()


def atomic_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch_save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(obj, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_jsonl_gz(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with gzip.open(tmp, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_jsonl_gz(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


class StopFlag:
    """SIGTERM/SIGUSR1 set a flag; the loop checkpoints at the next boundary and exits.
    The harness preamble keeps its own traps on the batch shell; this only affects Python."""

    def __init__(self):
        self.stop = False
        self.signum = None
        for s in (signal.SIGTERM, signal.SIGUSR1):
            signal.signal(s, self._handle)

    def _handle(self, signum, frame):
        self.stop, self.signum = True, signum


class CostLedger:
    """Append-only GPU-time ledger across job segments (never rolled back on resume).

    Each process start opens a new segment; its elapsed wall time is updated in place. The sum
    over segments charges loading, idle time, discarded work and preemption rework.
    """

    def __init__(self, path, kind):
        self.path = path
        self.t0 = time.time()
        try:
            self.data = json.load(open(path))
        except (OSError, json.JSONDecodeError):
            self.data = {"kind": kind, "segments": []}
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        self.seg = {"slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "restart": os.environ.get("SLURM_RESTART_COUNT", "0"), "host": socket.gethostname(),
                    "gpu": gpu, "start": self.t0, "elapsed_s": 0.0, "note": ""}
        self.data["segments"].append(self.seg)
        self.flush()

    def flush(self, note=None):
        self.seg["elapsed_s"] = time.time() - self.t0
        if note:
            self.seg["note"] = note
        atomic_json(self.path, self.data)

    def total_gpu_s(self):
        return sum(s["elapsed_s"] for s in self.data["segments"])


class Run:
    """Layout under $PROJ_DATA/runs/<run_id>/."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.id = cfg["run_id"]
        self.root = os.path.join(cfg["proj_data"], "runs", self.id)
        self.chash = config_hash(cfg)
        os.makedirs(self.root, exist_ok=True)
        cpath = os.path.join(self.root, "config.json")
        if os.path.exists(cpath):
            old = json.load(open(cpath))
            if old["hash"] != self.chash:
                raise SystemExit(f"config hash mismatch for run {self.id}: {old['hash']} != {self.chash}. "
                                 "Use a new run_id; never resume a run with a different configuration.")
        else:
            atomic_json(cpath, {"hash": self.chash, "config": cfg})

    def path(self, *p):
        return os.path.join(self.root, *p)

    def done_ok(self):
        try:
            d = json.load(open(self.path("DONE.json")))
        except (OSError, json.JSONDecodeError):
            return False
        return d.get("config_hash") == self.chash and all(
            os.path.exists(self.path(a["path"])) for a in d.get("artifacts", []))
