"""`bc2rl analyze`: amendment-A2 analysis (protocol/AMENDMENT-A2 Sections 4-7).

Reads attempt-level records under $PROJ_DATA and writes tables, JSON and PNG figures:
  audit.csv        per panel x input count: success, format/parse failures, illegal operand (input reuse or
                   unavailable number), arithmetic errors, truncation, success-count histogram, mixed-group
                   frequency at G=4 (pass@4 - pass^4 estimator)
  curves.csv       pass@k and pass^k, k = 1..n, by panel and input count (equal-weight overall)
  contrasts.csv    paired difference curves with pointwise and simultaneous 95% bootstrap bands
  retention.csv    7/8-split retention (initially consistent subgroup), k = 1..8, by stratum
  transitions.csv  initial class (first 7 attempts) x final success count
  coverage.csv     exploratory X1: coverage loss (initial c > 0 -> final 0) and gain vs a no-change reference
  e4_gate.json     E4 gate on validation
  mech.csv/json    E2 per-pair contrasts and checkpoint summaries
Test panels are read only when `allow_test` is true (after the result set is frozen).
Missing panels (runs not finished) are listed, never imputed.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

from . import metrics
from .runtime import read_jsonl_gz

KMAX = 15
COLORS = {"grpo": "#2a78d6", "filtered": "#eb6834", "initial": "#1baf7a", "H": "#2a78d6", "L": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ----------------------------------------------------------------------------- loading

def panel_path(pd, p):
    if p.get("path"):
        return os.path.join(pd, p["path"])
    return os.path.join(pd, "evals", p["eval_id"], f"{p['tag']}.jsonl.gz")


def load_panel(pd, p):
    recs = read_jsonl_gz(panel_path(pd, p))
    by = defaultdict(dict)
    h, errs = {}, defaultdict(Counter)
    for r in recs:
        by[r["pid"]][int(r["attempt"])] = int(r["reward"])
        h[r["pid"]] = int(r["n_inputs"])
    pids = sorted(by)
    n = min(len(v) for v in by.values())
    att = np.array([[by[q][a] for a in range(n)] for q in pids], dtype=int)
    return {"spec": p, "pids": pids, "h": np.array([h[q] for q in pids]), "att": att, "recs": recs}


def curves_for(att, h, kmax):
    """Per-problem pass@k / pass^k arrays for k = 1..kmax from attempt matrix (problems x n)."""
    n = att.shape[1]
    c = att.sum(1)
    out = {}
    for k in range(1, min(kmax, n) + 1):
        out[("pass@", k)] = np.array([metrics.pass_at_k(n, int(x), k) for x in c])
        out[("pass^", k)] = np.array([metrics.pass_hat_k(n, int(x), k) for x in c])
    return out


def weighted(v, h):
    return metrics.horizon_weighted(v, h)


# ----------------------------------------------------------------------------- audit

ERR_GROUPS = {"parse_or_format": ("malformed", "no_ops"), "illegal_operand": ("illegal_operand",),
              "arith_error": ("arith_error",), "too_few_ops": ("too_few_ops",), "wrong_final": ("wrong_final",),
              "truncated": ("truncated",), "success": ("success",)}


def audit_rows(name, pan):
    rows = []
    recs = pan["recs"]
    for hh in sorted(set(pan["h"].tolist())):
        rs = [r for r in recs if int(r["n_inputs"]) == hh]
        ec = Counter(r["error"] for r in rs)
        att = pan["att"][pan["h"] == hh]
        n = att.shape[1]
        c = att.sum(1)
        row = {"panel": name, "n_inputs": hh, "problems": len(c), "attempts": n, "pass@1": float(c.mean() / n)}
        for g, keys in ERR_GROUPS.items():
            row[f"frac_{g}"] = sum(ec[k] for k in keys) / max(len(rs), 1)
        row["format_valid"] = float(np.mean([r.get("format_valid", False) for r in rs]))
        if n >= 4:
            row["mixed_G4"] = float(np.mean([metrics.pass_at_k(n, int(x), 4) - metrics.pass_hat_k(n, int(x), 4) for x in c]))
        row["problems_any_success"] = int((c > 0).sum())
        row["problems_all_success"] = int((c == n).sum())
        row["success_count_hist"] = json.dumps(dict(sorted(Counter(int(x) for x in c).items())))
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------- bootstrap

def strat_indices(h, n_boot, seed=0):
    return metrics.bootstrap_indices(h, n_boot, seed)


def boot_w(v, idx):
    return metrics.boot_weighted(v, idx)


def contrast(a, b, h, n_boot, kmax, seed=0):
    """Paired whole-problem bootstrap of weighted(a) - weighted(b) curves; simultaneous band per family."""
    idx = strat_indices(h, n_boot, seed)
    ca, cb = curves_for(a, h, kmax), curves_for(b, h, kmax)
    rows = []
    for fam in ("pass@", "pass^"):
        ks = sorted(k for f, k in ca if f == fam and (f, k) in cb)
        pts, bs = [], []
        for k in ks:
            d = ca[(fam, k)] - cb[(fam, k)]
            pts.append(weighted(d, h)[0])
            bs.append(boot_w(d, idx))
        pts, bs = np.array(pts), np.array(bs)                    # (K,), (K, B)
        maxdev = np.abs(bs - pts[:, None]).max(0)
        c95 = float(np.percentile(maxdev, 95))
        for j, k in enumerate(ks):
            lo, hi = np.percentile(bs[j], [2.5, 97.5])
            byh = weighted(ca[(fam, k)] - cb[(fam, k)], h)[1]
            rows.append({"family": fam, "k": k, "diff": float(pts[j]), "pw_lo": float(lo), "pw_hi": float(hi),
                         "sim_lo": float(pts[j] - c95), "sim_hi": float(pts[j] + c95),
                         **{f"diff_h{hh}": float(v) for hh, v in byh.items()}})
    return rows


def rare_bound(N, k_grid=14, strata=3, alpha=0.05):
    return metrics.zero_event_upper_bound(N, alpha / (k_grid * strata))


# ----------------------------------------------------------------------------- retention

def retention(init, final, n_boot, min_n=12, seed=0, kmax=8):
    """Initial attempts 0-6 define consistency (>= 6/7); attempts 7-14 are the independent baseline (n = 8);
    final uses all fresh attempts. Problems are aligned by pid."""
    pos = {p: i for i, p in enumerate(final["pids"])}
    ia, fa, h = init["att"], final["att"][[pos[p] for p in init["pids"]]], init["h"]
    if ia.shape[1] < 15:
        raise ValueError("retention needs a 15-attempt initial panel")
    c7 = ia[:, :7].sum(1)
    base = ia[:, 7:15]
    S = c7 >= 6
    rows, trans = [], []
    rng = np.random.default_rng(seed)
    for hh in sorted(set(h.tolist())) + ["all"]:
        m = S & ((h == hh) if hh != "all" else np.ones_like(S))
        nS = int(m.sum())
        row = {"n_inputs": hh, "n_consistent": nS, "status": "ok" if nS >= min_n else ("underpowered" if nS else "none")}
        if nS:
            bc, fc = base[m].sum(1), fa[m].sum(1)
            for k in range(1, kmax + 1):
                for fam, fn in (("pass@", metrics.pass_at_k), ("pass^", metrics.pass_hat_k)):
                    b = np.array([fn(8, int(x), k) for x in bc])
                    f = np.array([fn(fa.shape[1], int(x), k) for x in fc])
                    d = f - b
                    bs = d[rng.integers(0, nS, size=(n_boot, nS))].mean(1)
                    lo, hi = np.percentile(bs, [2.5, 97.5])
                    row[f"{fam}{k}_base"], row[f"{fam}{k}_final"] = float(b.mean()), float(f.mean())
                    row[f"{fam}{k}_diff"], row[f"{fam}{k}_lo"], row[f"{fam}{k}_hi"] = float(d.mean()), float(lo), float(hi)
            row["final_trunc_or_parse_frac"] = None
        rows.append(row)
    cls = np.where(c7 == 0, "0/7", np.where(c7 >= 6, "6-7/7", "1-5/7"))
    fcnt = fa.sum(1)
    nf = fa.shape[1]
    fbin = np.where(fcnt == 0, "0", np.where(fcnt == nf, f"{nf}", np.where(fcnt >= nf // 2 + 1, f"{nf // 2 + 1}-{nf - 1}",
                                                                           f"1-{nf // 2}")))
    for hh in sorted(set(h.tolist())):
        cnt = Counter(zip(cls[h == hh], fbin[h == hh]))
        for (a, b), v in sorted(cnt.items()):
            trans.append({"n_inputs": hh, "initial_class": a, "final_bin": b, "problems": v})
    return rows, trans, {"S": S, "base_c": base.sum(1), "final_c": fcnt, "h": h, "c7": c7}


def _p_zero(c, n, m):
    """No-change reference: P(0 successes in m fresh attempts | c of n initial), Jeffreys posterior predictive."""
    a, b = c + 0.5, n - c + 0.5
    lbeta = lambda x, y: math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)
    return math.exp(lbeta(a, b + m) - lbeta(a, b))


def coverage(init, final, n_sim=10000, seed=0):
    """Exploratory X1 (DECISIONS 2026-10-25 16:00): per stratum, problems lost (initial c > 0 of all initial attempts,
    final 0) and gained (initial 0, final > 0), each against its no-change expectation and a 95% simulation band."""
    pos = {p: i for i, p in enumerate(final["pids"])}
    ia, fa, h = init["att"], final["att"][[pos[p] for p in init["pids"]]], init["h"]
    n, m = ia.shape[1], fa.shape[1]
    ic, fc = ia.sum(1), fa.sum(1)
    p0 = np.array([_p_zero(int(c), n, m) for c in ic])
    rng = np.random.default_rng(seed)
    rows = []
    for hh in sorted(set(h.tolist())) + ["all"]:
        sel = (h == hh) if hh != "all" else np.ones(len(h), bool)
        row = {"n_inputs": hh}
        for kind, msk, pe, obs in (("lost", sel & (ic > 0), p0, fc == 0), ("gained", sel & (ic == 0), 1 - p0, fc > 0)):
            sims = (rng.random((n_sim, int(msk.sum()))) < pe[msk]).sum(1)
            lo, hi = np.percentile(sims, [2.5, 97.5]) if msk.any() else (0.0, 0.0)
            row.update({f"{kind}_n": int(msk.sum()), kind: int((obs & msk).sum()), f"{kind}_expected": float(pe[msk].sum()),
                        f"{kind}_lo": float(lo), f"{kind}_hi": float(hi)})
        rows.append(row)
    return rows


def failure_mix(final, S_pids):
    rs = [r for r in final["recs"] if r["pid"] in S_pids]
    ec = Counter(r["error"] for r in rs)
    return {k: v / max(len(rs), 1) for k, v in ec.items()}


# ----------------------------------------------------------------------------- E2

def mech_analysis(pd, run_ids, n_boot):
    rows, summ = [], {}
    for rid in run_ids:
        sp = os.path.join(pd, "runs", rid, "summary.json")
        if not os.path.exists(sp):
            summ[rid] = {"status": "missing"}
            continue
        s = json.load(open(sp))
        fk = {}
        for t in s["forks"]:
            fk[t] = json.load(open(os.path.join(pd, "runs", rid, "forks", f"{t}.json")))
        b = fk["base"]
        hb = np.array(b["probe"]["h"])
        cb = np.array(b["probe"]["c"])
        n = int(b["probe"]["n"][0])
        idx = strat_indices(hb, n_boot, 0)
        pair_rows = []
        for p in s["pairs_used"]:
            i = p["pair"]
            H, L = fk[f"p{i}-H"], fk[f"p{i}-L"]
            assert H["probe"]["pids"] == L["probe"]["pids"] == b["probe"]["pids"]
            cH, cL = np.array(H["probe"]["c"]), np.array(L["probe"]["c"])
            r = {"run": rid, "pair": i, "opposite_sign": p["opposite_sign"], "align_H": p["exact"]["H"]["align"],
                 "align_L": p["exact"]["L"]["align"], "d_norm": p["exact"]["H"]["d_norm_equal"], "rho": s["rho"],
                 "len_H": p["len_H"], "len_L": p["len_L"], "composition": json.dumps(p["composition"])}
            for arm, f in (("H", H), ("L", L)):
                r[f"dfit_{arm}"] = f["anchorfit"]["mean_norm_ll"] - b["anchorfit"]["mean_norm_ll"]
                r[f"dcheck_{arm}"] = f["anchorcheck"]["mean_norm_ll"] - b["anchorcheck"]["mean_norm_ll"]
                r[f"dcheck_num_{arm}"] = f["anchorcheck"]["num_ll_per_token"] - b["anchorcheck"]["num_ll_per_token"]
                r[f"dcheck_delim_{arm}"] = f["anchorcheck"]["delim_ll_per_token"] - b["anchorcheck"]["delim_ll_per_token"]
                r[f"kl_{arm}"] = f["kl_median"]
                r[f"taylor_ratio_{arm}"] = (f["taylor_actual_fit"] / f["taylor_pred_fit"]) if f["taylor_pred_fit"] else None
                r[f"probe_p1_{arm}"] = weighted(np.array(f["probe"]["c"]) / n, hb)[0]
                r[f"format_valid_{arm}"] = f["probe"]["format_valid"]
            r["check_contrast"] = r["dcheck_H"] - r["dcheck_L"]
            r["fit_contrast"] = r["dfit_H"] - r["dfit_L"]
            d = (cH - cL) / n
            bs = boot_w(d, idx)
            r["probe_p1_contrast"] = weighted(d, hb)[0]
            r["probe_p1_lo"], r["probe_p1_hi"] = [float(x) for x in np.percentile(bs, [2.5, 97.5])]
            for k in (1, 2, 4, 8, 15):
                a = np.array([metrics.pass_hat_k(n, int(x), k) for x in cH]) - np.array([metrics.pass_hat_k(n, int(x), k) for x in cL])
                r[f"probe_passhat{k}_contrast"] = weighted(a, hb)[0]
                a = np.array([metrics.pass_at_k(n, int(x), k) for x in cH]) - np.array([metrics.pass_at_k(n, int(x), k) for x in cL])
                r[f"probe_passat{k}_contrast"] = weighted(a, hb)[0]
            pair_rows.append(r)
        rows += pair_rows
        cc = np.array([r["check_contrast"] for r in pair_rows])
        pc = np.array([r["probe_p1_contrast"] for r in pair_rows])
        P = len(cc)
        signs = np.array(list(itertools.product([1, -1], repeat=P))) if P else np.zeros((0, 0))
        def signflip(x):
            if P == 0:
                return None
            obs = abs(x.mean())
            return float(np.mean(np.abs((signs * x).mean(1)) >= obs - 1e-15))
        summ[rid] = {"status": "ok", "pairs": P, "rho": s["rho"], "pool_stats": s["selection"]["pool_stats"],
                     "n_pools": s["selection"]["n_pools"],
                     "pairs_status": Counter(p["status"] for p in s["selection"]["pairs"]),
                     "check_contrast_mean": float(cc.mean()) if P else None,
                     "check_contrast_positive": int((cc > 0).sum()),
                     "check_signflip_p": signflip(cc), "probe_p1_contrast_mean": float(pc.mean()) if P else None,
                     "probe_signflip_p": signflip(pc), "opposite_sign_pairs": int(sum(r["opposite_sign"] for r in pair_rows)),
                     "base_probe_p1": weighted(cb / n, hb)[0],
                     "base_probe_p1_by_inputs": weighted(cb / n, hb)[1]}
    return rows, summ


# ----------------------------------------------------------------------------- figures

def _style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def _kaxis(ax):
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("k")


INIT_LS = {"light": "--", "extended": "-", "contbc": ":"}
INIT_LABEL = {"light": "light BC (16 steps)", "extended": "extended BC (128 steps)", "contbc": "continued BC (256 steps)"}
METHOD_LABEL = {"initial": "initial BC checkpoint", "grpo": "after GRPO (r256)", "filtered": "after filtered (r256)"}


def figures(out, panels, curves_rows, contrast_sets, ret_sets, mech_rows, toy_csv, traj):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlecolor": INK, "axes.labelcolor": INK, "lines.linewidth": 2})
    fd = os.path.join(out, "figures")
    os.makedirs(fd, exist_ok=True)
    made = []

    # F1 initial and final k-curves by input count (for each split separately)
    by_split = defaultdict(list)
    for name, pan in panels.items():
        by_split[pan["spec"].get("split", "?")].append(name)
    for split, names in by_split.items():
        for fam, lab in (("pass@", "pass@k"), ("pass^", "pass^k")):
            fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
            used_m, used_i = set(), set()
            for ax, hh in zip(axes, (3, 5, 10)):
                for name in names:
                    sp = panels[name]["spec"]
                    rows = [r for r in curves_rows if r["panel"] == name and r["family"] == fam]
                    if not rows or sp.get("figure") is False:
                        continue
                    m, i = sp.get("method", "initial"), sp.get("init", "extended")
                    used_m.add(m)
                    used_i.add(i)
                    ax.plot([r["k"] for r in rows], [r[f"h{hh}"] for r in rows], color=COLORS.get(m, MUTED),
                            ls=INIT_LS.get(i, "-"), lw=1.2, alpha=0.8)
                ax.set_title(f"{hh} inputs", fontsize=9)
                _kaxis(ax)
                _style(ax)
            axes[0].set_ylabel(lab)
            from matplotlib.lines import Line2D
            ms = [m for m in METHOD_LABEL if m in used_m]
            ins = [i for i in INIT_LS if i in used_i]
            hd = [Line2D([], [], color=COLORS[m]) for m in ms] + [Line2D([], [], color=MUTED, ls=INIT_LS[i]) for i in ins]
            if not hd:                     # every panel of this split has figure: false
                plt.close(fig)
                continue
            fig.legend(hd, [METHOD_LABEL[m] for m in ms] + [INIT_LABEL[i] for i in ins],
                       loc="lower center", ncol=len(hd), frameon=False, fontsize=8)
            fig.suptitle(f"{lab} by input count ({split} panels; one line per checkpoint/seed)", fontsize=10)
            fig.tight_layout(rect=(0, 0.08, 1, 0.95))
            fn = f"curves_{split}_{'passat' if fam == 'pass@' else 'passhat'}.png"
            fig.savefig(os.path.join(fd, fn), dpi=150)
            plt.close(fig)
            made.append(fn)

    # F2 contrast curves with simultaneous bands
    for setname, cs in contrast_sets.items():
        if not cs:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        for ax, fam in zip(axes, ("pass@", "pass^")):
            for name, rows in cs.items():
                rr = [r for r in rows if r["family"] == fam]
                if not rr:
                    continue
                k = [r["k"] for r in rr]
                ls = INIT_LS["light"] if "light" in name else INIT_LS["extended"]
                ax.plot(k, [r["diff"] for r in rr], color=INK, ls=ls, lw=1.4, alpha=0.9)
                ax.fill_between(k, [r["sim_lo"] for r in rr], [r["sim_hi"] for r in rr], color=INK, alpha=0.06, lw=0)
            ax.axhline(0, color=MUTED, lw=0.8)
            _kaxis(ax)
            ax.set_ylabel(f"difference in {fam}k")
            _style(ax)
        from matplotlib.lines import Line2D
        fig.legend([Line2D([], [], color=INK, ls=INIT_LS["light"]), Line2D([], [], color=INK, ls=INIT_LS["extended"])],
                   ["light-BC init (one line per seed)", "extended-BC init (one line per seed)"], loc="lower center",
                   ncol=2, frameon=False, fontsize=8)
        fig.suptitle(f"{setname}: paired difference curves; shaded = simultaneous 95% band over k", fontsize=10)
        fig.tight_layout(rect=(0, 0.08, 1, 0.93))
        fn = f"contrast_{setname}.png"
        fig.savefig(os.path.join(fd, fn), dpi=150)
        plt.close(fig)
        made.append(fn)

    # F3 retention: per-problem baseline (8) vs final (15) success rate on consistent problems
    for setname, rs in ret_sets.items():
        if not rs:
            continue
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        for name, d in rs.items():
            S = d["S"]
            if not S.any():
                continue
            col = COLORS["grpo"] if "grpo" in name else COLORS["filtered"]
            mk = "o" if "extended" in name else "s"
            jitter = (np.random.default_rng(0).random(S.sum()) - 0.5) * 0.02
            ax.scatter(d["base_c"][S] / 8 + jitter, d["final_c"][S] / 15 + jitter, s=14, color=col, marker=mk, alpha=0.6,
                       edgecolors="white", linewidths=0.5)
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8)
        ax.set_xlabel("independent initial baseline success rate (attempts 8-15)")
        ax.set_ylabel("final success rate (15 fresh attempts)")
        _style(ax)
        from matplotlib.lines import Line2D
        ax.legend([Line2D([], [], color=COLORS["grpo"], marker="o", ls=""), Line2D([], [], color=COLORS["filtered"], marker="o", ls=""),
                   Line2D([], [], color=MUTED, marker="o", ls=""), Line2D([], [], color=MUTED, marker="s", ls="")],
                  ["GRPO", "filtered", "extended BC init", "light BC init"], frameon=False, fontsize=8, loc="lower left")
        ax.set_title(f"Initially consistent problems (>= 6/7): {setname}", fontsize=9)
        fig.tight_layout()
        fn = f"retention_{setname}.png"
        fig.savefig(os.path.join(fd, fn), dpi=150)
        plt.close(fig)
        made.append(fn)

    # F4 E2 paired contrasts
    if mech_rows:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        runs = sorted({r["run"] for r in mech_rows})
        for j, rid in enumerate(runs):
            rr = [r for r in mech_rows if r["run"] == rid]
            for r in rr:
                axes[0].plot([j - 0.12, j + 0.12], [r["dcheck_L"], r["dcheck_H"]], color=MUTED, lw=0.8)
                axes[0].scatter([j - 0.12], [r["dcheck_L"]], color=COLORS["L"], s=22, zorder=3)
                axes[0].scatter([j + 0.12], [r["dcheck_H"]], color=COLORS["H"], s=22, zorder=3)
                axes[1].scatter([r["align_H"] - r["align_L"]], [r["check_contrast"]], color=INK, s=18)
                axes[2].errorbar([j], [r["probe_p1_contrast"]], yerr=[[r["probe_p1_contrast"] - r["probe_p1_lo"]],
                                                                       [r["probe_p1_hi"] - r["probe_p1_contrast"]]],
                                 fmt="o", color=INK, ms=4, lw=1, alpha=0.7)
        for ax in axes:
            _style(ax)
        axes[0].set_xticks(range(len(runs)))
        axes[0].set_xticklabels([r.replace("bc2rl-e2-", "") for r in runs], rotation=30, fontsize=7)
        axes[0].set_ylabel("change in anchor-check log-lik.\n(per problem, /256)")
        axes[0].set_title("low (orange) vs high (blue) alignment arm", fontsize=9)
        axes[1].set_xlabel("alignment contrast cos(h,d_H) - cos(h,d_L)")
        axes[1].set_ylabel("anchor-check contrast (H - L)")
        axes[1].axhline(0, color=MUTED, lw=0.8)
        axes[2].set_xticks(range(len(runs)))
        axes[2].set_xticklabels([r.replace("bc2rl-e2-", "") for r in runs], rotation=30, fontsize=7)
        axes[2].axhline(0, color=MUTED, lw=0.8)
        axes[2].set_ylabel("probe pass@1 contrast H - L\n(95% problem bootstrap)")
        fig.suptitle("E2 matched-update pairs (each line/point = one pair)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(os.path.join(fd, "e2_pairs.png"), dpi=150)
        plt.close(fig)
        made.append("e2_pairs.png")

    # F5 E0 toy curves
    if toy_csv and os.path.exists(toy_csv):
        rows = list(csv.DictReader(open(toy_csv)))
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        th = [float(r["theta"]) for r in rows if r["model"] == "opposing"]
        p = np.linspace(0, 1, 101)
        for G, ls in ((2, ":"), (4, "-"), (8, "--")):       # method colors are reserved for GRPO / filtered
            axes[0].plot(p, 1 - p ** G - (1 - p) ** G, color=INK, ls=ls, label=f"G={G}")
        axes[0].set_xlabel("success probability p")
        axes[0].set_ylabel("P(mixed group)")
        axes[0].legend(frameon=False, fontsize=8)
        for k, ls in ((1, "-"), (4, "--"), (8, ":")):
            axes[1].plot(th, [float(r[f"C{k}"]) for r in rows if r["model"] == "opposing"], color=INK, ls=ls, label=f"k={k}")
            axes[2].plot(th, [float(r[f"R{k}"]) for r in rows if r["model"] == "opposing"], color=INK, ls=ls, label=f"k={k}")
        axes[1].set_title("opposing features: pass@k (coverage)", fontsize=9)
        axes[2].set_title("opposing features: pass^k (repeated success)", fontsize=9)
        for ax in axes[1:]:
            ax.set_xlabel("theta")
            ax.legend(frameon=False, fontsize=8)
        for ax in axes:
            _style(ax)
        fig.suptitle("E0 analytic illustration (exact model, not measured results)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(os.path.join(fd, "e0_toy.png"), dpi=150)
        plt.close(fig)
        made.append("e0_toy.png")

    # F6 signal opportunity vs improvement along E3 trajectories
    if traj:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for name, t in traj.items():
            col = COLORS[t["method"]]
            ls = INIT_LS.get(t["init"], "-")
            r = np.array(t["rounds"])
            mg = np.array(t["mixed"], float)
            w = 16
            sm = np.convolve(mg, np.ones(w) / w, mode="valid")
            axes[0].plot(r[w - 1:], sm, color=col, ls=ls, lw=1.2, alpha=0.8)
            if t.get("val"):
                vr = sorted(t["val"])
                axes[1].plot(vr, [t["val"][x] for x in vr], color=col, ls=ls, lw=1.2, marker="o", ms=3, alpha=0.8)
        axes[0].set_xlabel("rollout round")
        axes[0].set_ylabel("mixed groups per round (of 32; 16-round mean)")
        axes[1].set_xlabel("rollout round")
        axes[1].set_ylabel("validation pass@1 (144 x 4 attempts)")
        for ax in axes:
            _style(ax)
        from matplotlib.lines import Line2D
        fig.legend([Line2D([], [], color=COLORS["grpo"]), Line2D([], [], color=COLORS["filtered"]),
                    Line2D([], [], color=MUTED, ls="--"), Line2D([], [], color=MUTED)],
                   ["GRPO", "filtered", "light BC init", "extended BC init"], loc="lower center", ncol=4, frameon=False, fontsize=8)
        fig.suptitle("E3 trajectories: learning-signal availability and validation success", fontsize=10)
        fig.tight_layout(rect=(0, 0.08, 1, 0.93))
        fig.savefig(os.path.join(fd, "e3_trajectories.png"), dpi=150)
        plt.close(fig)
        made.append("e3_trajectories.png")
    return made


# ----------------------------------------------------------------------------- command

def write_csv(path, rows):
    if not rows:
        open(path, "w").close()
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def cmd_analyze(cfg):
    an = cfg["analysis"]
    pd = cfg["proj_data"]
    out = cfg.get("_results_dir") or os.path.join(pd, "analysis", cfg["run_id"])
    os.makedirs(out, exist_ok=True)
    n_boot = an.get("n_boot", 10000)
    panels, missing = {}, []
    for p in an["panels"]:
        if p.get("split") == "test" and not an.get("allow_test"):
            continue
        if not os.path.exists(panel_path(pd, p)):
            missing.append(p["name"])
            continue
        panels[p["name"]] = load_panel(pd, p)
    numbers = {"missing_panels": missing, "panels": {}}
    audit, curves_rows = [], []
    for name, pan in panels.items():
        audit += audit_rows(name, pan)
        cv = curves_for(pan["att"], pan["h"], KMAX)
        for (fam, k), v in sorted(cv.items()):
            ov, byh = weighted(v, pan["h"])
            curves_rows.append({"panel": name, "family": fam, "k": k, "overall": ov, **{f"h{hh}": x for hh, x in byh.items()}})
        numbers["panels"][name] = {"spec": pan["spec"], "n_problems": len(pan["pids"]), "attempts": int(pan["att"].shape[1])}
    write_csv(os.path.join(out, "audit.csv"), audit)
    write_csv(os.path.join(out, "curves.csv"), curves_rows)
    # zero-event bounds where every problem has pass^k = 0
    rare = []
    for name, pan in panels.items():
        n = pan["att"].shape[1]
        for hh in sorted(set(pan["h"].tolist())):
            att = pan["att"][pan["h"] == hh]
            for k in range(2, min(n, KMAX) + 1):
                ind = att[:, :k].all(1)
                if not ind.any():
                    rare.append({"panel": name, "n_inputs": hh, "k": k, "N": len(att), "upper_bound": rare_bound(len(att))})
    write_csv(os.path.join(out, "rare_events.csv"), rare)
    # contrasts
    contrast_rows, contrast_sets = [], defaultdict(dict)
    for c in an.get("contrasts", []):
        if c["a"] not in panels or c["b"] not in panels:
            numbers.setdefault("contrasts_missing", []).append(c["name"])
            continue
        A, B = panels[c["a"]], panels[c["b"]]
        if A["pids"] != B["pids"]:
            raise SystemExit(f"contrast {c['name']}: panels differ")
        k = min(A["att"].shape[1], B["att"].shape[1])
        rows = contrast(A["att"][:, :k], B["att"][:, :k], A["h"], n_boot, KMAX)
        for r in rows:
            r["contrast"] = c["name"]
        contrast_rows += rows
        contrast_sets[c.get("set", "contrasts")][c["name"]] = rows
    write_csv(os.path.join(out, "contrasts.csv"), contrast_rows)
    # retention
    ret_rows, trans_rows, ret_sets = [], [], defaultdict(dict)
    for r in an.get("retention", []):
        if r["initial"] not in panels or r["final"] not in panels:
            numbers.setdefault("retention_missing", []).append(r["name"])
            continue
        rows, trans, d = retention(panels[r["initial"]], panels[r["final"]], n_boot)
        S_p = {p for p, s in zip(panels[r["initial"]]["pids"], d["S"]) if s}
        for x in rows:
            x.update(name=r["name"], split=r.get("split"), init=r.get("init"), method=r.get("method"), seed=r.get("seed"))
        rows[-1]["final_failure_mix_on_S"] = json.dumps(failure_mix(panels[r["final"]], S_p))
        for x in trans:
            x.update(name=r["name"])
        ret_rows += rows
        trans_rows += trans
        ret_sets[r.get("set", "retention")][r["name"]] = d
    write_csv(os.path.join(out, "retention.csv"), ret_rows)
    cov_rows = []
    for r in an.get("retention", []):
        if r["initial"] in panels and r["final"] in panels:
            for x in coverage(panels[r["initial"]], panels[r["final"]], n_sim=n_boot):
                x.update(name=r["name"], split=r.get("split"), init=r.get("init"), method=r.get("method"), seed=r.get("seed"))
                cov_rows.append(x)
    write_csv(os.path.join(out, "coverage.csv"), cov_rows)
    write_csv(os.path.join(out, "transitions.csv"), trans_rows)
    # E4 gate (validation)
    gate = {"rule": "final val pass@1 on initially consistent subset falls >= 5 pp vs independent baseline in >= 2 "
                    "extended-BC GRPO seeds, each with >= 12 qualifying problems", "seeds": {}}
    for x in ret_rows:
        if x.get("split") == "val" and x.get("init") == "extended" and x.get("method") == "grpo" and x["n_inputs"] == "all":
            gate["seeds"][str(x["seed"])] = {"n_consistent": x["n_consistent"], "pass@1_diff": x.get("pass@1_diff"),
                                             "meets": bool(x["n_consistent"] >= 12 and (x.get("pass@1_diff") or 0) <= -0.05),
                                             "final_failure_mix_on_S": x.get("final_failure_mix_on_S")}
    gate["seeds_meeting"] = sum(v["meets"] for v in gate["seeds"].values())
    gate["gate_met_on_numbers"] = gate["seeds_meeting"] >= 2
    json.dump(gate, open(os.path.join(out, "e4_gate.json"), "w"), indent=1, default=str)
    # E2
    mech_rows, mech_summ = mech_analysis(pd, an.get("mech_runs", []), n_boot)
    write_csv(os.path.join(out, "mech.csv"), mech_rows)
    json.dump(mech_summ, open(os.path.join(out, "mech_summary.json"), "w"), indent=1, default=str)
    # E3 trajectories (training-side logs; no test data)
    traj = {}
    for t in an.get("trajectories", []):
        mp = os.path.join(pd, "runs", t["run_id"], "metrics.jsonl")
        sp = os.path.join(pd, "runs", t["run_id"], "summary.json")
        if not os.path.exists(sp):
            continue
        rows = [json.loads(l) for l in open(mp) if l.strip()]
        s = json.load(open(sp))
        traj[t["run_id"]] = {"init": t["init"], "method": t["method"], "seed": t["seed"],
                             "rounds": [r["round"] for r in rows], "mixed": [r["mixed_groups"] for r in rows],
                             "successes": [r["successes"] for r in rows], "skipped": sum(r["skipped"] for r in rows),
                             "val": {c["round"]: c["val"]["metrics"]["pass@1"] for c in s["checkpoints"] if c.get("val")},
                             "val_by_inputs": {c["round"]: {h: v.get("pass@1") for h, v in c["val"]["by_inputs"].items()}
                                               for c in s["checkpoints"] if c.get("val")},
                             "opt_steps": s["counters"]["opt_steps"], "gpu_s": s["gpu_s_total"],
                             "gpus": sorted({g["gpu"] for g in s["segments"]}),
                             "mean_mixed": float(np.mean([r["mixed_groups"] for r in rows])),
                             "mean_mixed_first64": float(np.mean([r["mixed_groups"] for r in rows[:64]]))}
    json.dump(traj, open(os.path.join(out, "trajectories.json"), "w"), indent=1)
    made = figures(out, panels, curves_rows, contrast_sets, ret_sets, mech_rows, an.get("toy_csv") and os.path.join(pd, an["toy_csv"]), traj)
    numbers["figures"] = made
    numbers["allow_test"] = bool(an.get("allow_test"))
    json.dump(numbers, open(os.path.join(out, "numbers.json"), "w"), indent=1, default=str)
    print(json.dumps({"panels": len(panels), "missing": missing, "figures": made}))
    return 0
