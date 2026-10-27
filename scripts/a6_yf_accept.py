#!/usr/bin/env python3
"""A6 structural acceptance of the yf (confirmation) evaluations before look 1. No outcome statistics are read.

For each of the 24 initial and 72 final tags of `y_look1.required()` and each panel (yf3, yf5):
  summary file exists; tag and panel fields agree; one row per problem (row count = panel size, unique pids);
  every row has n = attempts (48 initial, 32 final) and 0 <= c <= n; completions = rows x attempts;
  protocol: attempts = n, panel_sha256 the same for all tags of the panel, cap_ext present;
  adapter_sha256 = configs/a6/adapter_sha256.json when the adapter is registered there, and the same on both panels;
  the attempt-record file exists.
Writes yf_accept.json (one line per tag and panel) to --out; exit status 1 if any check fails.
Usage: a6_yf_accept.py --evals DIR --sha configs/a6/adapter_sha256.json --out DIR [--rows yf3=768,yf5=128]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from y_look1 import required  # noqa: E402


def check(evals, sha_map, rows):
    inits, fins = required()
    out, panel_sha = [], {}
    for t in inits + fins:
        n_want = 48 if t in inits else 32
        shas = set()
        for panel, nr in rows.items():
            p = os.path.join(evals, f"{t}.{panel}.summary.json")
            row = {"tag": t, "panel": panel, "problems": []}
            if not os.path.exists(p):
                row["problems"].append("summary missing")
                out.append(row)
                continue
            s = json.load(open(p))
            c = s.get("counts", [])
            pr = s.get("protocol", {})
            bad = row["problems"]
            if s.get("tag") != t or s.get("panel") != panel:
                bad.append("tag/panel field")
            if len(c) != nr or len({r["pid"] for r in c}) != nr:
                bad.append(f"rows {len(c)} (unique {len({r['pid'] for r in c})}) != {nr}")
            if any(int(r["n"]) != n_want or not 0 <= int(r["c"]) <= n_want for r in c):
                bad.append("row n or c out of range")
            if s.get("completions") != nr * n_want:
                bad.append(f"completions {s.get('completions')} != {nr * n_want}")
            if pr.get("attempts") != n_want or "cap_ext" not in pr:
                bad.append("protocol attempts/cap_ext")
            panel_sha.setdefault(panel, set()).add(pr.get("panel_sha256"))
            a = s.get("adapter")
            if a in sha_map and s.get("adapter_sha256") != sha_map[a]:
                bad.append("adapter sha != registered")
            row["registered"] = a in sha_map
            shas.add(s.get("adapter_sha256"))
            if not os.path.exists(os.path.join(evals, f"{t}.{panel}.jsonl.gz")):
                bad.append("records missing")
            out.append(row)
        if len(shas) > 1:
            out.append({"tag": t, "panel": "both", "problems": ["adapter sha differs between panels"]})
    for panel, v in panel_sha.items():
        if len(v) != 1:
            out.append({"tag": "*", "panel": panel, "problems": [f"{len(v)} panel sha256 values"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default="yf3=768,yf5=128")
    a = ap.parse_args()
    rows = {k: int(v) for k, v in (x.split("=") for x in a.rows.split(","))}
    out = check(a.evals, json.load(open(a.sha)), rows)
    failed = [r for r in out if r["problems"]]
    os.makedirs(a.out, exist_ok=True)
    json.dump({"checked": len(out), "failed": len(failed), "registered": sum(bool(r.get("registered")) for r in out),
               "rows": out}, open(os.path.join(a.out, "yf_accept.json"), "w"), indent=1)
    print(f"[yf-accept] checked={len(out)} failed={len(failed)}")
    for r in failed[:20]:
        print(f"[yf-accept] FAIL {r['tag']}.{r['panel']}: {r['problems']}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
