"""Figure and table for the E3 anchor audit (`scripts/e3_anchor_audit.py` output; descriptive, mechanism splits only).

Writes `anchor_audit.csv` (one row per branch: r0 and r256 anchor-check / anchor-fit mean normalized log-likelihood, the
change, and the change by input count and token class) and `figures/e3_anchor_ll.png` (anchor-check trajectory by round).
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from bc2rl.analysis import COLORS, INIT_LABEL, INK, MUTED, _style  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = json.load(open(args.audit))["rows"]
    by = {(r["branch"], r["round"]): r for r in rows}
    branches = sorted({r["branch"] for r in rows})
    table = []
    for b in branches:
        init, method, seed = b.split("-")
        r0, r1 = by[(b, 0)], by[(b, 256)]
        c0, c1 = r0["anchorcheck"], r1["anchorcheck"]
        row = {"branch": b, "init": init, "method": method, "seed": seed,
               "check_r0": c0["mean_norm_ll"], "check_r256": c1["mean_norm_ll"],
               "check_delta": c1["mean_norm_ll"] - c0["mean_norm_ll"],
               "fit_delta": r1["anchorfit"]["mean_norm_ll"] - r0["anchorfit"]["mean_norm_ll"],
               "num_tok_delta": c1["num_ll_per_token"] - c0["num_ll_per_token"],
               "delim_tok_delta": c1["delim_ll_per_token"] - c0["delim_ll_per_token"]}
        for h in c0["by_inputs"]:
            row[f"check_delta_h{h}"] = c1["by_inputs"][h] - c0["by_inputs"][h]
        table.append(row)
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    with open(os.path.join(args.out, "anchor_audit.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 9, "axes.titlecolor": INK, "axes.labelcolor": INK, "lines.linewidth": 2})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, init in zip(axes, ("light", "extended")):
        for b in branches:
            if not b.startswith(init + "-"):
                continue
            rs = sorted(rd for (bb, rd) in by if bb == b)
            ax.plot(rs, [by[(b, rd)]["anchorcheck"]["mean_norm_ll"] for rd in rs], color=COLORS[b.split("-")[1]],
                    marker="o", ms=4, lw=1.4, alpha=0.85)
        ax.set_title(INIT_LABEL[init], fontsize=9)
        ax.set_xlabel("online round")
        ax.set_xticks([0, 64, 128, 256])
        _style(ax)
    axes[0].set_ylabel("anchor-check mean normalized log-lik.")
    fig.legend([Line2D([], [], color=COLORS["grpo"]), Line2D([], [], color=COLORS["filtered"])],
               ["GRPO", "filtered"], loc="lower center", ncol=2, frameon=False, fontsize=8, labelcolor=MUTED)
    fig.suptitle("Reference-trace log-likelihood on held-out anchor-check problems (fp32; one line per seed)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(os.path.join(args.out, "figures", "e3_anchor_ll.png"), dpi=150)
    print(f"[anchor] {len(table)} branches -> anchor_audit.csv, figures/e3_anchor_ll.png")


if __name__ == "__main__":
    main()
