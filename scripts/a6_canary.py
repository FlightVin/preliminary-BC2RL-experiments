"""Amendment A6 canary check (CPU): for every Y3 run with metrics, the per-round `used_by_inputs` and `by_inputs`.
MASK must use only 3-input completions (used 5 = used 10 = 0) while its stream still samples 5- and 10-input prompts
(by_inputs 5 / 10 not NaN). T3 must sample only 3-input prompts (by_inputs 5 / 10 NaN)."""
import argparse
import glob
import json
import math
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = {}
    for path in sorted(glob.glob(os.path.join(a.runs, "bc2rl-y3-*", "metrics.jsonl"))):
        run = os.path.basename(os.path.dirname(path))
        rows = [json.loads(x) for x in open(path) if x.strip()]
        rows = [r for r in rows if "used_by_inputs" in r]
        used = {h: sum(r["used_by_inputs"][h] for r in rows) for h in ("3", "5", "10")}
        sampled = {h: sum(not math.isnan(r["by_inputs"][h]) for r in rows) for h in ("3", "5", "10")}
        arm = run.split("-")[-2]
        if arm == "mask":
            ok = used["5"] == 0 and used["10"] == 0 and sampled["5"] > 0 and sampled["10"] > 0
        elif arm == "t3":
            ok = sampled["5"] == 0 and sampled["10"] == 0 and used["3"] > 0
        else:
            ok = None
        rep[run] = {"rounds": len(rows), "used_by_inputs": used, "rounds_sampling": sampled, "ok": ok}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "a6_canary.json"), "w") as f:
        json.dump(rep, f, indent=1)
    bad = [k for k, v in rep.items() if v["ok"] is False]
    print(f"[a6-canary] runs {len(rep)}; failed {bad}")


if __name__ == "__main__":
    main()
