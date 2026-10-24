"""`bc2rl report`: tables, paired bootstrap, rare-event bounds, discovery split, curves, cost, figures.

Input: cfg["report"]["matrix"] — a JSON file (frozen before test evaluation) listing
  conditions: [{tag, eval_id, init, method, seed, endpoint, train_run, bc_run, bc_steps, online_rounds}]
    init in {nobc, light, extended}; method in {initial, grpo, filtered, contbc};
    endpoint in {initial, online, total, bc256}
  train_runs: [{run_id, kind: bc|online, init, method, seed}]
Output (to --results-dir): metrics_long.csv, paired.csv, rankings.json, discovery.csv, curves.csv,
cost.csv, rare_events.csv, figures/*.png, numbers.json. Everything is recomputed from the attempt
records under $PROJ_DATA; nothing is read from earlier summaries.
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import numpy as np

from . import metrics
from .evaluate import counts_by_problem, load_eval_records

METRIC_NAMES = ("pass@1", "pass@4", "pass^4", "pass^2", "mixed4")
INITS = ("nobc", "light", "extended")
METHODS = ("grpo", "filtered")


def _per_problem(recs):
    pids, n, c, h = counts_by_problem(recs)
    return pids, n, c, h


def load_conditions(pd, matrix):
    data = {}
    ref_pids = None
    for cnd in matrix["conditions"]:
        recs = load_eval_records(pd, cnd["eval_id"], cnd["tag"])
        pids, n, c, h = _per_problem(recs)
        if ref_pids is None:
            ref_pids, ref_h = pids, h
        if pids != ref_pids:
            raise SystemExit(f"{cnd['tag']}: problem set differs from the reference panel")
        by_attempt = defaultdict(dict)
        for r in recs:
            by_attempt[r["pid"]][r["attempt"]] = r["reward"]
        data[cnd["tag"]] = {"cond": cnd, "n": n, "c": c, "recs": recs,
                            "att": np.array([[by_attempt[p][a] for a in range(int(n.max()))] for p in pids])}
    return data, ref_pids, ref_h


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def cmd_report(cfg):
    rp = cfg["report"]
    pd = cfg["proj_data"]
    out = cfg.get("_results_dir") or os.path.join(pd, "reports", cfg["run_id"])
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    matrix = json.load(open(rp["matrix"] if os.path.isabs(rp["matrix"]) else os.path.join(os.getcwd(), rp["matrix"])))
    data, pids, H = load_conditions(pd, matrix)
    hs = sorted(set(H.tolist()))
    n_boot = rp.get("n_boot", 10000)
    idx = metrics.bootstrap_indices(H, n_boot, seed=rp.get("boot_seed", 0))
    pp = {t: {m: metrics.per_problem(d["n"], d["c"], m) for m in METRIC_NAMES} for t, d in data.items()}
    boots = {t: {m: metrics.boot_weighted(pp[t][m], idx) for m in METRIC_NAMES} for t in data}
    numbers = {"n_problems": len(pids), "horizons": hs, "n_boot": n_boot, "conditions": {}}

    # ---- 1. metrics by condition and horizon, with task-sampling CIs
    rows = []
    for t, d in data.items():
        cnd = d["cond"]
        numbers["conditions"][t] = {"cond": cnd, "metrics": {}}
        for m in METRIC_NAMES:
            overall, byh = metrics.horizon_weighted(pp[t][m], H)
            lo, hi = np.percentile(boots[t][m], [2.5, 97.5])
            numbers["conditions"][t]["metrics"][m] = {"all": overall, "lo": float(lo), "hi": float(hi),
                                                      **{str(h): v for h, v in byh.items()}}
            rows.append([t, cnd["init"], cnd["method"], cnd.get("seed"), cnd["endpoint"], "all", m,
                         round(overall, 6), round(float(lo), 6), round(float(hi), 6)])
            for h, v in byh.items():
                rows.append([t, cnd["init"], cnd["method"], cnd.get("seed"), cnd["endpoint"], h, m, round(v, 6), "", ""])
    write_csv(os.path.join(out, "metrics_long.csv"),
              ["tag", "init", "method", "seed", "endpoint", "horizon", "metric", "value", "lo95", "hi95"], rows)

    def find(init, method, seed, endpoint):
        for t, d in data.items():
            c = d["cond"]
            if c["init"] == init and c["method"] == method and c["endpoint"] == endpoint and \
                    (c.get("seed") == seed or (c["method"] == "initial" and c["init"] == "nobc")):
                return t
        return None

    # ---- 2. paired differences: GRPO - filtered (per seed and seed-mean), final - initial, vs continued BC
    prow = []

    def diff(name, a_tags, b_tags, m):
        """Seed-averaged per-problem difference (a list vs b list), paired bootstrap over problems."""
        a = np.mean([pp[t][m] for t in a_tags], axis=0)
        b = np.mean([pp[t][m] for t in b_tags], axis=0)
        pt = metrics.horizon_weighted(a - b, H)[0]
        bs = np.mean([boots[t][m] for t in a_tags], axis=0) - np.mean([boots[t][m] for t in b_tags], axis=0)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        byh = metrics.horizon_weighted(a - b, H)[1]
        prow.append([name, m, round(pt, 6), round(float(lo), 6), round(float(hi), 6)] +
                    [round(byh[h], 6) for h in hs])
        return pt

    seeds = sorted({d["cond"]["seed"] for d in data.values() if d["cond"].get("seed") is not None})
    for ep in ("online", "total"):
        for init in INITS:
            g = [find(init, "grpo", s, ep) for s in seeds]
            f = [find(init, "filtered", s, ep) for s in seeds]
            for m in METRIC_NAMES:
                for s, gt, ft in zip(seeds, g, f):
                    if gt and ft:
                        diff(f"{ep}:{init}:grpo-filtered:s{s}", [gt], [ft], m)
                if all(g) and all(f):
                    diff(f"{ep}:{init}:grpo-filtered:mean", g, f, m)
            for method in METHODS:
                fin = [find(init, method, s, ep) for s in seeds]
                ini = [find(init, "initial", s, "initial") for s in seeds]
                if all(fin) and all(ini):
                    for m in METRIC_NAMES:
                        diff(f"{ep}:{init}:{method}-initial:mean", fin, ini, m)
        for method in METHODS:
            fin = [find("extended", method, s, ep) for s in seeds]
            cb = [find("extended", "contbc", s, "bc256") for s in seeds]
            if all(fin) and all(cb):
                for m in METRIC_NAMES:
                    diff(f"{ep}:extended:{method}-contbc:mean", fin, cb, m)
    write_csv(os.path.join(out, "paired.csv"), ["comparison", "metric", "diff", "lo95", "hi95"] +
              [f"h{h}" for h in hs], prow)

    # ---- 3. rankings (H1, H3): initial pass@1 vs final pass@1 by method and endpoint
    def mean_metric(tags, m="pass@1"):
        return float(np.mean([numbers["conditions"][t]["metrics"][m]["all"] for t in tags]))

    rankings = {}
    init_score = {i: mean_metric([find(i, "initial", s, "initial") for s in seeds]) for i in INITS
                  if all(find(i, "initial", s, "initial") for s in seeds)}
    rankings["initial_pass@1"] = init_score
    for ep in ("online", "total"):
        for method in METHODS:
            sc = {}
            for i in INITS:
                tags = [find(i, method, s, ep) for s in seeds]
                if all(tags):
                    sc[i] = {m: mean_metric(tags, m) for m in ("pass@1", "pass@4", "pass^4")}
            rankings[f"{ep}:{method}"] = sc
    for k, v in list(rankings.items()):
        if k != "initial_pass@1" and v:
            rankings[k + ":order_pass@1"] = sorted(v, key=lambda i: -v[i]["pass@1"])
    rankings["initial:order_pass@1"] = sorted(init_score, key=lambda i: -init_score[i])
    json.dump(rankings, open(os.path.join(out, "rankings.json"), "w"), indent=1)

    # ---- 4. rare events: preselected quartet (attempts 0-3) when a condition-horizon shows zero pass^4
    rrows = []
    for t, d in data.items():
        for h in hs:
            sel = H == h
            ph4 = pp[t]["pass^4"][sel]
            q = np.array([metrics.quartet_indicator(a) for a in d["att"][sel]])
            ub = metrics.zero_event_upper_bound(int(sel.sum())) if q.sum() == 0 else None
            rrows.append([t, h, int(sel.sum()), round(float(ph4.mean()), 6), int(q.sum()),
                          round(ub, 6) if ub is not None else "", int((d["c"][sel] >= 4).sum())])
    write_csv(os.path.join(out, "rare_events.csv"),
              ["tag", "horizon", "problems", "pass^4", "quartet_successes", "upper95_if_zero", "problems_c>=4"], rrows)

    # ---- 5. discovery / consolidation / retention (initial attempts 0-6 group, 7-14 baseline)
    drows = []
    for t, d in data.items():
        cnd = d["cond"]
        if cnd["method"] not in METHODS:
            continue
        it = find(cnd["init"], "initial", cnd.get("seed"), "initial")
        if not it:
            continue
        att0 = data[it]["att"]
        g7 = att0[:, :7].sum(1)
        groups = np.where(g7 == 0, "none", np.where(g7 == 7, "all", "intermittent"))
        base8 = att0[:, 7:15].mean(1)
        fin = d["att"].mean(1)
        for h in hs + ["all"]:
            for gname in ("none", "intermittent", "all"):
                sel = (groups == gname) & ((H == h) if h != "all" else True)
                if sel.sum() == 0:
                    continue
                drows.append([t, cnd["init"], cnd["method"], cnd.get("seed"), cnd["endpoint"], h, gname,
                              int(sel.sum()), round(float(base8[sel].mean()), 6), round(float(fin[sel].mean()), 6)])
    write_csv(os.path.join(out, "discovery.csv"), ["tag", "init", "method", "seed", "endpoint", "horizon", "group",
                                                   "problems", "initial_acc_attempts7to14", "final_acc"], drows)

    # ---- 6. training curves and compute ledger from run summaries
    crow, cost = [], []
    for tr in matrix.get("train_runs", []):
        sp = os.path.join(pd, "runs", tr["run_id"], "summary.json")
        if not os.path.exists(sp):
            cost.append([tr["run_id"], tr["kind"], tr.get("init"), tr.get("method"), tr.get("seed"), "missing",
                         "", "", "", "", ""])
            continue
        s = json.load(open(sp))
        gpus = sorted({seg["gpu"] for seg in s["segments"]})
        cost.append([tr["run_id"], tr["kind"], tr.get("init"), tr.get("method"), tr.get("seed"), "complete",
                     round(s["gpu_s_total"] / 3600, 4), len(s["segments"]), ";".join(gpus),
                     s.get("counters", {}).get("generated_tokens", ""), s.get("counters", {}).get("opt_steps",
                                                                                                   s.get("steps", ""))])
        mpath = os.path.join(pd, "runs", tr["run_id"], "metrics.jsonl")
        for line in open(mpath):
            r = json.loads(line)
            if tr["kind"] == "online":
                crow.append([tr["run_id"], tr.get("init"), tr.get("method"), tr.get("seed"), r["round"],
                             r["reward_mean"], r["mixed_groups"], r["all_success_groups"], r["all_fail_groups"],
                             int(r["skipped"]), r["entropy"], r["mean_len"], r["grad_norm"] if r["grad_norm"] is not None else "",
                             r.get("twopass_absdev_mean", r["ratio_max_dev"]), r["format_valid"], r["truncated"], r["gen_tokens"]])
        for c in s.get("checkpoints", []):
            if c.get("val"):
                vm = c["val"]["metrics"]
                crow.append([tr["run_id"], tr.get("init"), tr.get("method"), tr.get("seed"),
                             f"val@{c.get('round', c.get('step'))}", vm.get("pass@1"), vm.get("mixed4"), "", "", "",
                             c.get("heldout_bc_loss", ""), "", "", "", "", "", c.get("gpu_s_so_far", "")])
    write_csv(os.path.join(out, "curves.csv"),
              ["run_id", "init", "method", "seed", "round", "reward_or_val_pass@1", "mixed_or_val_mixed4", "all_success",
               "all_fail", "skipped", "entropy_or_heldout_loss", "mean_len", "grad_norm", "twopass_absdev_mean",
               "format_valid", "truncated", "gen_tokens_or_gpu_s"], crow)
    write_csv(os.path.join(out, "cost.csv"), ["run_id", "kind", "init", "method", "seed", "status", "gpu_hours",
                                              "segments", "gpus", "generated_tokens", "opt_steps"], cost)

    # ---- 7. figures
    try:
        _figures(out, data, numbers, rankings, crow, seeds, find, hs)
    except Exception as e:  # figures never block the tables
        numbers["figure_error"] = repr(e)
    json.dump(numbers, open(os.path.join(out, "numbers.json"), "w"), indent=1, default=str)
    print(f"[report] wrote {out}")
    return 0


def _figures(out, data, numbers, rankings, crow, seeds, find, hs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"nobc": "#4C72B0", "light": "#DD8452", "extended": "#55A868"}
    marks = {"initial": "o", "grpo": "^", "filtered": "s", "contbc": "D"}
    # coverage-reliability per horizon
    fig, axes = plt.subplots(1, len(hs), figsize=(4.2 * len(hs), 3.8), sharey=False)
    for ax, h in zip(np.atleast_1d(axes), hs):
        for t, d in data.items():
            c = d["cond"]
            if c["endpoint"] not in ("initial", "online", "bc256"):
                continue
            m = numbers["conditions"][t]["metrics"]
            ax.scatter(m["pass@4"][str(h)], m["pass^4"][str(h)], c=colors[c["init"]], marker=marks[c["method"]],
                       s=36, alpha=0.8, edgecolors="k", linewidths=0.3)
        ax.set_title(f"{h} inputs")
        ax.set_xlabel("pass@4")
        ax.set_ylabel("pass^4")
    handles = [plt.Line2D([], [], color=v, marker="o", ls="") for v in colors.values()] + \
              [plt.Line2D([], [], color="gray", marker=v, ls="") for v in marks.values()]
    fig.legend(handles, list(colors) + list(marks), loc="lower center", ncol=7, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(out, "figures", "coverage_reliability.png"), dpi=150)
    plt.close(fig)
    # initial -> final pass@1 per init/method (equal online rounds), seeds shown
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, init in enumerate(INITS):
        for j, method in enumerate(METHODS):
            for s in seeds:
                it, ft = find(init, "initial", s, "initial"), find(init, method, s, "online")
                if it and ft:
                    a = numbers["conditions"][it]["metrics"]["pass@1"]["all"]
                    b = numbers["conditions"][ft]["metrics"]["pass@1"]["all"]
                    x = i * 3 + j
                    ax.plot([x - 0.3, x + 0.3], [a, b], color=colors[init], marker=marks[method], alpha=0.8)
    ax.set_xticks([i * 3 + j for i in range(3) for j in range(2)])
    ax.set_xticklabels([f"{i}\n{m}" for i in INITS for m in METHODS], fontsize=7)
    ax.set_ylabel("test pass@1 (initial -> equal-round endpoint)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "initial_to_final.png"), dpi=150)
    plt.close(fig)
    # training curves: validation pass@1 vs round
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, method in zip(axes, METHODS):
        series = defaultdict(list)
        for r in crow:
            if r[2] == method and isinstance(r[4], str) and r[4].startswith("val@"):
                series[(r[1], r[3])].append((int(r[4][4:]), r[5]))
        for (init, s), pts in series.items():
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colors.get(init, "k"), marker=".", alpha=0.7)
        ax.set_title(method)
        ax.set_xlabel("online round")
    axes[0].set_ylabel("validation pass@1 (4 attempts)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "val_curves.png"), dpi=150)
    plt.close(fig)
    # mixed-group fraction per round
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, method in zip(axes, METHODS):
        series = defaultdict(list)
        for r in crow:
            if r[2] == method and isinstance(r[4], int):
                series[(r[1], r[3])].append((r[4], r[6] / 32))
        for (init, s), pts in series.items():
            pts.sort()
            xs = np.array([p[0] for p in pts])
            ys = np.convolve([p[1] for p in pts], np.ones(8) / 8, mode="same")
            ax.plot(xs, ys, color=colors.get(init, "k"), alpha=0.7)
        ax.set_title(f"{method}: mixed-group fraction (8-round mean)")
        ax.set_xlabel("online round")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "mixed_groups.png"), dpi=150)
    plt.close(fig)
