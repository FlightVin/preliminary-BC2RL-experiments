"""Pre-registered confirmation analysis X5 (protocol/AMENDMENT-A5-2026-10-26.md Section 3); runs once on the complete
fresh-panel evaluation. Refuses to run on an incomplete set of adapters.

Writes to --out: x5_primary.json (primary result + decision + pre-registered sensitivities), x5_bins.csv,
x5_curves.csv (Delta(K) for loss and consistency, K = 1..24, simultaneous bands), x5_full.csv / x5_contrasts.csv
(whole-panel pass@k / pass^k and final - initial contrasts, both panels), x5_consistent.csv, x5_5input.csv, figures/.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from bc2rl import xanalysis as X  # noqa: E402


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", required=True, help="$PROJ_DATA/evals/ext/<data name>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    panels = {}
    for name in ("xc3", "xc5"):
        counts = X.load_counts(a.evals, name)
        missing = [t for t in X.required_tags() if t not in counts]
        if missing:
            raise SystemExit(f"{name}: missing adapters {missing}; analysis refused")
        panels[name] = X.Panel({t: counts[t] for t in X.required_tags()})
        print(f"[x5] {name}: {len(panels[name].pids)} problems, {len(counts)} adapters", flush=True)
    p3 = panels["xc3"]
    res = {"primary": X.primary(p3, n_boot=a.n_boot)}
    print(f"[x5] PRIMARY {json.dumps({k: res['primary'][k] for k in ('mean_delta', 'ci95', 'decision', 'pairs_positive')})}",
          flush=True)
    sens = {
        "swap_roles": dict(swap=True),
        "K4": dict(K=4), "K8": dict(K=8), "K24": dict(K=24),
        "fine_bins": dict(bins=tuple((c, c) for c in range(1, 12))),
        "with_high_bin": dict(bins=X.PRIMARY_BINS + X.HIGH_BIN),
        "consistency_rel16": dict(measure="rel"),
    }
    res["sensitivity"] = {k: X.primary(p3, n_boot=a.n_boot, **v) for k, v in sens.items()}
    # matched-baseline restriction: primary bins whose mean baseline rate differs by <= 0.02 between inits in every seed
    rows = X.per_bin_table(p3)
    keep = []
    for j, (lo, hi) in enumerate(X.PRIMARY_BINS):
        ok = True
        for s in X.SEEDS:
            r = {x["init"]: x for x in rows if x["seed"] == s and x["bin"] == f"{lo}-{hi}"}
            if r["light"]["base_rate"] is None or r["extended"]["base_rate"] is None or \
                    abs(r["light"]["base_rate"] - r["extended"]["base_rate"]) > 0.02:
                ok = False
        if ok:
            keep.append((lo, hi))
    res["sensitivity"]["baseline_matched_bins"] = (X.primary(p3, n_boot=a.n_boot, bins=tuple(keep)) if keep
                                                   else {"status": "no bin met the 0.02 baseline-rate rule"})
    res["sensitivity"]["baseline_matched_bins_kept"] = keep
    for k, v in res["sensitivity"].items():
        if isinstance(v, dict) and "mean_delta" in v:
            print(f"[x5] {k}: {v['mean_delta']:+.4f} {v['ci95']} {v['decision']}", flush=True)
    write_csv(os.path.join(a.out, "x5_bins.csv"), rows)
    idx3 = X.boot_idx(len(p3.pids), a.n_boot, 1)
    curves = [dict(measure="loss", **r) for r in X.curve_family(p3, X.N_BASE, "loss", idx3)] + \
             [dict(measure="rel", **r) for r in X.curve_family(p3, X.N_BASE, "rel", idx3)]
    write_csv(os.path.join(a.out, "x5_curves.csv"), curves)
    write_csv(os.path.join(a.out, "x5_consistent.csv"), X.consistent_subgroup(p3))
    full, contr = [], []
    for name, p in panels.items():
        f, c = X.full_panel(p, idx=X.boot_idx(len(p.pids), a.n_boot, 2))
        full += [dict(panel=name, **r) for r in f]
        contr += [dict(panel=name, **r) for r in c]
    write_csv(os.path.join(a.out, "x5_full.csv"), full)
    write_csv(os.path.join(a.out, "x5_contrasts.csv"), contr)
    # 5-input coverage change next to the 3-input excess loss, per branch (descriptive association)
    assoc = []
    prim = X.comparison(p3)
    p5 = panels["xc5"]
    for i in X.INITS:
        for m in X.METHODS:
            for s in X.SEEDS:
                tf, ti = X.final_tag(i, m, s), X.init_tag(i, s)
                row = {"branch": tf, "E3_excess_loss_K16": float(prim["E"][(i, m, s)])}
                for k in (1, 4, 16, 32):
                    for nm, p in (("3", p3), ("5", p5)):
                        row[f"dC{k}_h{nm}"] = float((1 - X.loss_k(X.N_FIN, p.c[tf], k)).mean()
                                                    - (1 - X.loss_k(48, p.c[ti], k)).mean())
                assoc.append(row)
    write_csv(os.path.join(a.out, "x5_5input.csv"), assoc)
    res["n_problems"] = {k: len(v.pids) for k, v in panels.items()}
    res["n_boot"] = a.n_boot
    json.dump(res, open(os.path.join(a.out, "x5_primary.json"), "w"), indent=1, default=float)
    plots(a.out, rows, curves, contr)
    print("[x5] done", flush=True)


def plots(out, rows, curves, contr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"light": "#3b6ea8", "extended": "#c4572e"}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, m in zip(axes, X.METHODS):
        for i in X.INITS:
            for s in X.SEEDS:
                rr = [r for r in rows if r["init"] == i and r["seed"] == s and r["n"] >= X.MIN_BIN_N]
                ax.plot([r["base_rate"] for r in rr], [r[f"{m}_excess"] for r in rr], marker="o", color=col[i],
                        alpha=0.8, lw=1.2, label=f"{i}" if s == 11 else None)
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_title(f"{m}: excess loss (K = 16) by selection bin", fontsize=9)
        ax.set_xlabel("mean baseline success rate in bin (independent attempts)")
    axes[0].set_ylabel("final - baseline  (1-p)^16 estimate")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "x5_excess_by_bin.png"), dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for ax, meas, lab in zip(axes, ("loss", "rel"), ("excess loss (1-p)^K", "consistency change p^K")):
        rr = [r for r in curves if r["measure"] == meas]
        k = [r["K"] for r in rr]
        ax.plot(k, [r["delta"] for r in rr], color="#222", lw=2)
        ax.fill_between(k, [r["sim_lo"] for r in rr], [r["sim_hi"] for r in rr], color="#222", alpha=0.12, lw=0)
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_title(f"extended - light: {lab}", fontsize=9)
        ax.set_xlabel("K")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "x5_delta_curves.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
