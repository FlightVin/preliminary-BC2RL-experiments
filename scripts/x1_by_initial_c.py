"""POST HOC (after the test read; amendment A4, DECISIONS 2026-10-25): X1 coverage loss stratified by initial success count.

The X1 no-change reference depends on a Jeffreys prior. This check needs no prior: within each initial success count c
(1..14 of 15), it compares the share of problems that the round-256 adapter fails 15/15 between light-BC and
extended-BC branches (pooled over seeds; per method and pooled). It also gives an indirect standardization: the losses
extended-BC would show at the light-BC per-c loss rates. Descriptive only; uses the frozen retention pairs of
`configs/analysis/final.yaml`.

Writes `x1_by_c.csv` (split, n_inputs, init, method, c, n, lost) and `x1_by_c_summary.json`.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.getcwd())
from bc2rl.analysis import load_panel  # noqa: E402
from bc2rl.runtime import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/analysis/final.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    pd, an = cfg["proj_data"], cfg["analysis"]
    specs = {p["name"]: p for p in an["panels"]}
    cache = {}

    def panel(name):
        if name not in cache:
            cache[name] = load_panel(pd, specs[name])
        return cache[name]

    cnt = defaultdict(lambda: [0, 0])          # (split, h, init, method, c) -> [n, lost]
    for r in an["retention"]:
        ini, fin = panel(r["initial"]), panel(r["final"])
        pos = {p: i for i, p in enumerate(fin["pids"])}
        ic = ini["att"].sum(1)
        fc = fin["att"][[pos[p] for p in ini["pids"]]].sum(1)
        for h, c, f in zip(ini["h"].tolist(), ic.tolist(), fc.tolist()):
            if c > 0:
                k = (r["split"], h, r["init"], r["method"], c)
                cnt[k][0] += 1
                cnt[k][1] += int(f == 0)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "x1_by_c.csv"), "w") as fh:
        fh.write("split,n_inputs,init,method,c,n,lost\n")
        for k in sorted(cnt):
            fh.write(",".join(map(str, k + tuple(cnt[k]))) + "\n")

    summ = {}
    for split in sorted({k[0] for k in cnt}):
        for h in sorted({k[1] for k in cnt}):
            for meth in ("grpo", "filtered", "pooled"):
                by = {i: defaultdict(lambda: [0, 0]) for i in ("light", "extended")}
                for (s, hh, i, m, c), (n, l) in cnt.items():
                    if s == split and hh == h and (meth == "pooled" or m == meth):
                        by[i][c][0] += n
                        by[i][c][1] += l
                ext, lig = by["extended"], by["light"]
                std = sum(ext[c][0] * lig[c][1] / lig[c][0] for c in ext if lig[c][0] > 0)
                obs = sum(ext[c][1] for c in ext if lig[c][0] > 0)
                summ[f"{split}/h{h}/{meth}"] = {
                    "extended_n": sum(v[0] for v in ext.values()), "extended_lost": sum(v[1] for v in ext.values()),
                    "light_n": sum(v[0] for v in lig.values()), "light_lost": sum(v[1] for v in lig.values()),
                    "extended_lost_in_shared_c": obs, "extended_expected_at_light_rates": std,
                    "by_c": {c: {"ext": ext[c], "light": lig[c]} for c in sorted(set(ext) | set(lig))}}
    json.dump(summ, open(os.path.join(args.out, "x1_by_c_summary.json"), "w"), indent=1)
    for k, v in summ.items():
        print(f"[x1c] {k}: ext {v['extended_lost']}/{v['extended_n']} light {v['light_lost']}/{v['light_n']} "
              f"ext-in-shared-c {v['extended_lost_in_shared_c']} vs {v['extended_expected_at_light_rates']:.1f} at light rates",
              flush=True)


if __name__ == "__main__":
    main()
