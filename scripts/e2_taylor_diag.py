"""E2 Taylor-check diagnostic (development data only; amendment A2 deviation D-E2-1).

Question: is the failed calibration Taylor check (actual / predicted anchor-fit change outside [0.5, 2] at rho = 1)
caused by bf16 measurement noise, by curvature at one optimizer step, or by an implementation error?

Uses the calibration checkpoint (dev light-BC, seed 7), the anchor-fit set and the 4 saved calibration displacements.
For precision in (bf16, fp32): repeat f(theta0) twice (determinism); compute h; for rho in RHOS and each displacement,
the actual change and h.d; the same for 4 random sign directions of equal norm (unaligned control). Fit
actual = a*rho + b*rho^2 over rho <= 4. rho < 1 is diagnostic only and is never used as an E2 radius.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from bc2rl import task  # noqa: E402
from bc2rl.data import load_data  # noqa: E402
from bc2rl.mech import (anchor_direction, anchor_ll, encode_ref, flat_params, load_mech, set_flat,  # noqa: E402
                        token_classes, trainable)
from bc2rl.modeling import load_adapter, load_policy, load_tokenizer, model_info  # noqa: E402
from bc2rl.runtime import load_config  # noqa: E402

RHOS = [0.25, 0.5, 1, 2, 4, 8]


def measure(model, params, theta0, h, dirs, seqs, cls, pad, Lmax, micro):
    f0a = anchor_ll(model, seqs, cls, pad, Lmax, micro)["mean_norm_ll"]
    f0b = anchor_ll(model, seqs, cls, pad, Lmax, micro)["mean_norm_ll"]
    out = {"f0": f0a, "f0_repeat_absdiff": abs(f0a - f0b), "dirs": {}}
    for name, d in dirs.items():
        rows = []
        for rho in RHOS:
            set_flat(params, theta0 + rho * d)
            f1 = anchor_ll(model, seqs, cls, pad, Lmax, micro)["mean_norm_ll"]
            set_flat(params, theta0)
            rows.append({"rho": rho, "pred": float(torch.dot(h, rho * d)), "actual": f1 - f0a})
        x = np.array([r["rho"] for r in rows if r["rho"] <= 4])
        y = np.array([r["actual"] for r in rows if r["rho"] <= 4])
        a, b = np.linalg.lstsq(np.stack([x, x ** 2], 1), y, rcond=None)[0]
        pred1 = float(torch.dot(h, d))
        out["dirs"][name] = {"rows": rows, "fit_linear_a": float(a), "fit_quad_b": float(b), "pred_at_1": pred1,
                             "fit_a_over_pred": float(a / pred1) if pred1 else None,
                             "ratio_at_1": next(r["actual"] / r["pred"] for r in rows if r["rho"] == 1)}
        print(f"[diag] {name}: pred1 {pred1:.3e} fit a {a:.3e} b {b:.3e} ratio@1 {out['dirs'][name]['ratio_at_1']:.3f}",
              flush=True)
    assert torch.equal(flat_params(params), theta0)
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
    mdata, _ = load_mech(cfg)
    info = model_info(pd)
    tok = load_tokenizer(info["path"])
    model = load_policy(info["path"], cfg["model"]["lora"], seed=0, device="cuda")
    asha = load_adapter(model, os.path.join(pd, me["adapter"]))
    model.eval()
    params = trainable(model)
    theta0 = flat_params(params).clone()
    seqs = [encode_ref(tok, p, fmt, cfg["prompt_variant"]) for p in mdata["anchorfit"]]
    cls = [token_classes(tok, c) for _, c in seqs]
    saved = torch.load(os.path.join(pd, "runs", cfg["run_id"], "disp.pt"), map_location="cpu", weights_only=False)
    dirs = {f"p{k[0]}-{k[1]}": v.cuda() for k, v in saved["disp"]}
    g = torch.Generator().manual_seed(12345)
    s = float(next(iter(dirs.values())).norm())
    for i in range(4):
        r = torch.sign(torch.randn(theta0.numel(), generator=g))
        dirs[f"rand{i}"] = (r * (s / float(r.norm()))).cuda()
    Lmax, micro, pad = me["cap"], me["micro"], tok.pad_token_id
    res = {"adapter_sha256": asha, "config": args.config, "rhos": RHOS, "n_anchor": len(seqs)}
    for prec in ("bf16", "fp32"):
        if prec == "fp32":
            model.float()
            params = trainable(model)
            set_flat(params, theta0)
            assert torch.equal(flat_params(params), theta0)
        h = anchor_direction(model, params, seqs, pad, Lmax, micro)
        res[prec] = measure(model, params, theta0, h, dirs, seqs, cls, pad, Lmax, micro)
        res[prec]["h_norm"] = float(h.norm())
        res[prec]["h_cos_to_bf16"] = None
        if prec == "bf16":
            h_bf16 = h.clone()
        else:
            res[prec]["h_cos_to_bf16"] = float(torch.dot(h, h_bf16) / (h.norm() * h_bf16.norm()))
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, "taylor_diag.json"), "w"), indent=1)
    print(json.dumps({p: {k: v["ratio_at_1"] for k, v in res[p]["dirs"].items()} for p in ("bf16", "fp32")}))


if __name__ == "__main__":
    main()
