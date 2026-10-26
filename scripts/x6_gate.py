"""X6 radius gate (protocol/AMENDMENT-A5 Section 4.2): the largest rho of the grid that passes V1-V5 on every
development calibration given. Writes a gate file that `bc2rl mech` reads through {"from_calibration": <path>}.

Usage: python scripts/x6_gate.py --cal <calibration.json> [--cal ...] --out <gate.json>
"""
import argparse
import json
import os


def gate(cals):
    grids = [set(c["grid"]) for c in cals]
    common = sorted(set.intersection(*grids))
    passing = [r for r in common if all(c["per_rho"][str(r)]["all_ok"] for c in cals)]
    chosen = max(passing) if passing else None
    return {"chosen_rho": chosen, "passing_common": passing, "grid": common,
            "checks": {"all_ok": chosen is not None, "n_calibrations": len(cals),
                       "per_calibration": {c["run_id"]: {str(r): {k: c["per_rho"][str(r)].get(k)
                                                                  for k in ("V1", "V2", "V3", "V4", "V5", "all_ok")}
                                                         for r in common} for c in cals}},
            "rule": "largest rho in the common grid passing V1-V5 on every development calibration (A5 Section 4.2)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cals = [json.load(open(p)) for p in a.cal]
    g = gate(cals)
    g["sources"] = a.cal
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(g, open(a.out, "w"), indent=1)
    print(f"[x6-gate] chosen rho = {g['chosen_rho']}; passing {g['passing_common']}")
    print(json.dumps(g["checks"]["per_calibration"]))


if __name__ == "__main__":
    main()
