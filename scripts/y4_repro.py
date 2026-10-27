"""Amendment A6 Section 3.2 reproduction check and a Y2 continuation sample (CPU, no model forward pass).

1. For each seed and step 16 / 64 / 128: sha256 of the re-run adapter (`runs/bc2rl-y4-bc-path-s<seed>`) against the
   original (`runs/bc2rl-s1-bc-broad-s<seed>`), and the relative L2 distance ||a - b|| / ||b|| over all tensors (fp32).
   Step 32 of the re-run is reported with its distances to steps 16 and 64 of the original path.
2. From the yd records: per tag, the number of attempts truncated at cap 256 and their outcome after continuation, and
   up to `--n-samples` continued texts (first 300 + last 200 characters) for a manual sanity check.
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
from collections import Counter

import torch
from safetensors.torch import load_file

SEEDS = (11, 22, 33, 44, 55, 66)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel_l2(a_path, b_path):
    a, b = load_file(a_path), load_file(b_path)
    if sorted(a) != sorted(b):
        return None
    num = sum(float(((a[k].float() - b[k].float()) ** 2).sum()) for k in b)
    den = sum(float((b[k].float() ** 2).sum()) for k in b)
    return (num / den) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--evals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=6)
    a = ap.parse_args()
    orig = lambda s, n: os.path.join(a.runs, f"bc2rl-s1-bc-broad-s{s}/adapters/step{n:04d}.safetensors")
    rerun = lambda s, n: os.path.join(a.runs, f"bc2rl-y4-bc-path-s{s}/adapters/step{n:04d}.safetensors")
    rep = {"adapters": [], "step32": [], "continuation": {}, "samples": []}
    for s in SEEDS:
        for n in (16, 64, 128):
            o, r = orig(s, n), rerun(s, n)
            row = {"seed": s, "step": n, "orig_exists": os.path.exists(o), "rerun_exists": os.path.exists(r)}
            if row["orig_exists"] and row["rerun_exists"]:
                row.update(sha_equal=sha(o) == sha(r), rel_l2=rel_l2(r, o))
            rep["adapters"].append(row)
        r32 = rerun(s, 32)
        if os.path.exists(r32):
            rep["step32"].append({"seed": s, "rel_l2_to_orig16": rel_l2(r32, orig(s, 16)),
                                  "rel_l2_to_orig64": rel_l2(r32, orig(s, 64))})
    n_left = a.n_samples
    for path in sorted(glob.glob(os.path.join(a.evals, "*.yd3.jsonl.gz"))):
        tag = os.path.basename(path).split(".")[0]
        c = Counter()
        with gzip.open(path, "rt") as f:
            for line in f:
                r = json.loads(line)
                if r["finish"] != "length":
                    continue
                c[(r["x_code"], r["x_finish"])] += 1
                if n_left > 0 and sum(c.values()) == 1:
                    t = r.get("x_text", "")
                    rep["samples"].append({"tag": tag, "pid": r["pid"], "x_code": r["x_code"],
                                           "x_n_tokens": r["x_n_tokens"], "head": t[:300], "tail": t[-200:]})
                    n_left -= 1
        rep["continuation"][tag] = {f"{k[0]}/{k[1]}": v for k, v in sorted(c.items())}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "y4_repro.json"), "w") as f:
        json.dump(rep, f, indent=1)
    eq = [x.get("sha_equal") for x in rep["adapters"]]
    print(f"[y4-repro] sha equal {sum(e is True for e in eq)}/{len(eq)}; "
          f"max rel_l2 {max((x.get('rel_l2') or 0) for x in rep['adapters']):.3g}")


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
