"""report on synthetic attempt records: files written, numbers recomputable by hand."""
import csv
import json
import os

import numpy as np

from bc2rl.report import cmd_report
from bc2rl.runtime import write_jsonl_gz


def make_recs(rate_by_h, seed):
    rng = np.random.default_rng(seed)
    recs = []
    for h, rate in rate_by_h.items():
        for i in range(6):
            for a in range(15):
                recs.append({"pid": f"t-{h}-{i}", "n_inputs": h, "attempt": a, "reward": int(rng.random() < rate),
                             "n_tokens": 10, "finish": "stop", "format_valid": True, "valid_prefix": 1, "error": "x"})
    return recs


def test_report_end_to_end(tmp_path):
    pd = tmp_path / "pd"
    conds = []
    tags = {"base": ("nobc", "initial", None, "initial")}
    for s in (11, 22):
        for init in ("light", "extended"):
            tags[f"{init}-s{s}"] = (init, "initial", s, "initial")
        for init in ("nobc", "light", "extended"):
            for m in ("grpo", "filtered"):
                for ep in ("online", "total"):
                    tags[f"{init}-{m}-{ep}-s{s}"] = (init, m, s, ep)
        tags[f"extended-contbc-s{s}"] = ("extended", "contbc", s, "bc256")
    for k, (tag, (init, m, s, ep)) in enumerate(tags.items()):
        rate = {3: 0.6, 5: 0.2 + 0.1 * (init == "light"), 10: 0.02}
        write_jsonl_gz(str(pd / "evals" / "ev" / f"{tag}.jsonl.gz"), make_recs(rate, k))
        conds.append({"tag": tag, "eval_id": "ev", "init": init, "method": m, "seed": s, "endpoint": ep})
    mpath = tmp_path / "matrix.json"
    json.dump({"conditions": conds, "train_runs": []}, open(mpath, "w"))
    out = tmp_path / "out"
    cmd_report({"run_id": "r", "proj_data": str(pd), "_results_dir": str(out),
                "report": {"matrix": str(mpath), "n_boot": 200}})
    for f in ("metrics_long.csv", "paired.csv", "rankings.json", "rare_events.csv", "discovery.csv",
              "numbers.json", "figures/coverage_reliability.png"):
        assert os.path.exists(out / f), f
    num = json.load(open(out / "numbers.json"))
    assert "figure_error" not in num
    # recompute pass@1 for one condition by hand: mean over horizons of mean c/n
    from bc2rl.evaluate import load_eval_records
    recs = load_eval_records(str(pd), "ev", "light-s11")
    byh = {}
    for r in recs:
        byh.setdefault(r["n_inputs"], []).append(r["reward"])
    hand = np.mean([np.mean(v) for v in byh.values()])
    assert abs(num["conditions"]["light-s11"]["metrics"]["pass@1"]["all"] - hand) < 1e-12
    rows = list(csv.DictReader(open(out / "paired.csv")))
    assert any(r["comparison"] == "online:light:grpo-filtered:mean" for r in rows)
