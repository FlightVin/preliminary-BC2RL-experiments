"""E3 per-round anchor audit (amendment A2 Section 10, secondary diagnostic; observational).

For each E3 branch and each saved adapter (round 0 = the initial BC adapter, rounds 64, 128, 256), measure the mean
normalized log-likelihood of the reference traces of the anchor-check set (primary) and the anchor-fit set, with an
fp32 forward pass (D-E2-1: the bf16 floor ~2.5e-4 is too coarse). Reports overall, per input count and per token class
(numeric vs delimiter). Uses mechanism splits only: no validation or test data.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from bc2rl.data import load_data  # noqa: E402
from bc2rl.mech import anchor_ll, encode_ref, load_mech, token_classes  # noqa: E402
from bc2rl.modeling import load_adapter, load_policy, load_tokenizer, model_info  # noqa: E402
from bc2rl.runtime import load_config  # noqa: E402

BRANCHES = [f"{i}-{m}-s{s}" for s in (11, 22, 33) for i in ("light", "extended") for m in ("grpo", "filtered")]
SETS = ("anchorcheck", "anchorfit")


def adapters_for(pd, branch):
    s = json.load(open(os.path.join(pd, "runs", f"bc2rl-s2-{branch}", "summary.json")))
    out = [(0, s["init"]["adapter"], s["init_sha256"])]
    out += [(c["round"], c["adapter"], None) for c in s["checkpoints"] if c.get("adapter")]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mech/calib.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    me, pd = cfg["mech"], cfg["proj_data"]
    torch.manual_seed(0)
    data, _ = load_data(cfg, ["dev"])
    fmt = data["format"][0]
    mdata, man = load_mech(cfg)
    info = model_info(pd)
    tok = load_tokenizer(info["path"])
    model = load_policy(info["path"], cfg["model"]["lora"], seed=0, device="cuda")
    model.float()
    model.eval()
    Lmax, micro, pad = me["cap"], me["micro"], tok.pad_token_id
    seqs = {k: [encode_ref(tok, p, fmt, cfg["prompt_variant"]) for p in mdata[k]] for k in SETS}
    cls = {k: [token_classes(tok, c) for _, c in seqs[k]] for k in SETS}
    hz = {k: np.array([p.n_inputs for p in mdata[k]]) for k in SETS}
    part = os.path.join(args.out, "anchor_audit.partial.jsonl")
    done = set()
    if os.path.exists(part):
        done = {(r["branch"], r["round"]) for r in map(json.loads, open(part))}
    first = True
    for b in BRANCHES:
        for rnd, rel, want in adapters_for(pd, b):
            if (b, rnd) in done:
                continue
            sha = load_adapter(model, os.path.join(pd, rel))
            if want is not None and sha != want:
                raise SystemExit(f"{b} r{rnd}: adapter sha {sha} != recorded {want}")
            row = {"branch": b, "round": rnd, "adapter": rel, "sha256": sha}
            for k in SETS:
                r = anchor_ll(model, seqs[k], cls[k], pad, Lmax, micro)
                per = np.array(r["per_problem"])
                row[k] = {"mean_norm_ll": r["mean_norm_ll"], "num_ll_per_token": r["num_ll_per_token"],
                          "delim_ll_per_token": r["delim_ll_per_token"],
                          "by_inputs": {int(h): float(per[hz[k] == h].mean()) for h in sorted(set(hz[k].tolist()))}}
                if first and k == "anchorcheck":     # determinism check of the fp32 measurement
                    row[k]["repeat_absdiff"] = abs(anchor_ll(model, seqs[k], cls[k], pad, Lmax, micro)["mean_norm_ll"]
                                                   - r["mean_norm_ll"])
            first = False
            with open(part, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[audit] {b} r{rnd} check {row['anchorcheck']['mean_norm_ll']:.5f} fit {row['anchorfit']['mean_norm_ll']:.5f}",
                  flush=True)
    rows = [json.loads(l) for l in open(part)]
    json.dump({"config": args.config, "mech_manifest": man.get("splits"), "precision": "fp32", "rows": rows},
              open(os.path.join(args.out, "anchor_audit.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
