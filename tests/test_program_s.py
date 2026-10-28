"""Tests for A6 program S stage S3 online options (protocol Section 13.4) on the tiny random Qwen2 of test_a6.py (CPU):
class pool from ext panels, per-class streams, class masking, per-class signal counts, class-gradient diagnostic."""
import json
import subprocess
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_a6 import PD, ROOT, env, needs_tok, online_cfg, run_cli, write  # noqa: E402,F401

import numpy as np  # noqa: E402

from bc2rl import task  # noqa: E402


def _ext(base, name, panels):
    from bc2rl.data import data_dir
    from bc2rl.extension import draw_fresh
    d = os.path.join(data_dir(base), "ext", name)
    os.makedirs(d, exist_ok=True)
    out, _ = draw_fresh(13, task.GenConfig(), panels, set())
    man = {"panels": {}}
    for p, probs in out.items():
        task.save_split(os.path.join(d, f"{p}.jsonl"), probs)
        man["panels"][p] = {"count": len(probs), "n_inputs": 3, "sha256": task.manifest_hash(probs)}
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"))


@needs_tok
def test_class_pool_stream_mask_and_diagnostic(env):
    tmp, base = env
    from bc2rl.modeling import file_sha256
    from bc2rl.runtime import read_jsonl_gz
    _ext(base, "s3t", {"trA": (3, 6), "trB": (3, 6)})
    pe = {"name": "s3t", "classes": {"A": ["trA", 6], "B": ["trB", 6]}}
    split = {"A": 3, "B": 3}
    runs = {"ab": dict(pool_ext=pe, class_per_round=split),
            "ab-all": dict(pool_ext=pe, class_per_round=split, train_classes=["A", "B"]),
            "ab-diag": dict(pool_ext=pe, class_per_round=split, class_grad_every=1),
            "ab-mb": dict(pool_ext=pe, class_per_round=split, train_classes=["A"]),
            "a": dict(pool_ext={"name": "s3t", "classes": {"A": ["trA", 6]}})}
    for rid, extra in runs.items():
        run_cli("online", write(tmp, "s3-" + rid, online_cfg(base, "s3-" + rid, **extra)))
    rd = lambda rid: os.path.join(base["proj_data"], "runs", "s3-" + rid)
    sha = {rid: file_sha256(os.path.join(rd(rid), "adapters", "round0003.safetensors")) for rid in ("ab", "ab-all", "ab-diag")}
    assert sha["ab"] == sha["ab-all"] == sha["ab-diag"]          # all-class mask and the diagnostic change nothing
    rows = {rid: [json.loads(l) for l in open(os.path.join(rd(rid), "metrics.jsonl"))] for rid in runs}
    assert all(r["by_class"]["B"]["used"] == 0 and r["by_class"]["B"]["used_tokens"] == 0 for r in rows["ab-mb"])
    assert sum(r["by_class"]["B"]["used"] for r in rows["ab"]) > 0
    assert [r["successes"] for r in rows["ab-mb"]][:1] == [r["successes"] for r in rows["ab"]][:1]
    for r in rows["ab"]:
        bc = r["by_class"]
        assert bc["A"]["prompts"] == bc["B"]["prompts"] == 3
        assert all(v["mixed"] + v["all_success"] + v["all_fail"] == 3 for v in bc.values())
        assert sum(v["used"] for v in bc.values()) == 2 * sum(v["mixed"] for v in bc.values())   # G = 2
    for rnd in (1, 2, 3):
        recs = read_jsonl_gz(os.path.join(rd("ab"), "records", f"round{rnd:04d}.jsonl.gz"))
        assert sum(x["pid"].startswith("trA-") for x in recs) == sum(x["pid"].startswith("trB-") for x in recs) == 6
        assert all(x["pid"].startswith("trA-") for x in read_jsonl_gz(
            os.path.join(rd("a"), "records", f"round{rnd:04d}.jsonl.gz")))
    cg = [r["class_grad"] for r in rows["ab-diag"] if "class_grad" in r]
    assert len(cg) == 3 and any("cos" in c for c in cg)
    assert all(-1.0001 <= c["cos"] <= 1.0001 for c in cg if "cos" in c)
    assert all("class_grad" not in r for r in rows["ab"])
    bad = online_cfg(base, "s3-bad", train_classes=["A"])
    run_cli("online", write(tmp, "s3-bad", bad), expect=1)


@needs_tok
def test_drift_kl_self_zero_stop_token_and_missing(env):
    """A6 stage R1 (protocol 14.1): reference sampling with log-probs, FP32 scoring, sequence KL per problem."""
    import hashlib
    import torch
    from bc2rl.drift import select_problems
    from bc2rl.extension import load_ext
    from bc2rl.modeling import eos_ids, load_policy, load_tokenizer, model_info, save_adapter
    tmp, base = env
    _ext(base, "drt", {"p3": (3, 6), "p5": (5, 4)})
    model = load_policy(base["model_path_override"], base["model"]["lora"], seed=0, device="cpu")
    ad0 = os.path.join(base["proj_data"], "drift-ad0.safetensors")
    save_adapter(model, ad0)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad and "lora_B" in n:
                p.add_(0.05 * torch.randn_like(p))
    ad1 = os.path.join(base["proj_data"], "drift-ad1.safetensors")
    save_adapter(model, ad1)
    dc = {"data_name": "drt", "select": {"p3": 4, "p5": 2}, "salt": "A6-R drift", "n_ref": 3, "cap": 6, "seed": 8,
          "batch_seqs": 8, "micro": 4, "out": "t",
          "refs": [{"tag": "r0", "adapter": ad0}],
          "score": [{"ref": "r0", "tag": "self", "adapter": ad0}, {"ref": "r0", "tag": "other", "adapter": ad1},
                    {"ref": "r0", "tag": "gone", "adapter": "nope.safetensors"}]}
    res = tmp / "dres"
    r = subprocess.run([sys.executable, "-m", "bc2rl", "drift", "--config", str(write(tmp, "drift", dict(base, run_id="d", drift=dc))),
                        "--results-dir", str(res)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    ext, _ = load_ext(base, "drt")
    sel = select_problems(ext, {"p3": 4, "p5": 2}, "A6-R drift")
    h = lambda p: hashlib.sha256(f"A6-R drift:{p.pid}".encode()).hexdigest()
    assert [p.pid for p in sel[:4]] == [p.pid for p in sorted(ext["p3"], key=h)[:4]]
    s_self = json.load(open(res / "drift_self.json"))
    s_oth = json.load(open(res / "drift_other.json"))
    assert len(s_self["per_problem"]["pid"]) == 6 and all(v == 0.0 for v in s_self["per_problem"]["kl"])
    assert all(np.isfinite(s_oth["per_problem"]["kl"])) and s_oth["kl_mean"] != 0.0
    ref = json.load(open(res / "drift_ref_r0.json"))
    assert abs(ref["check"]["signed_mean_per_token"]) < 0.05 and ref["summary"]["responses"] == 18
    st = torch.load(os.path.join(base["proj_data"], "drift", "t", "refs", "r0.pt"), map_location="cpu", weights_only=False)
    tok = load_tokenizer(model_info(base["proj_data"])["path"])
    stops = set(eos_ids(tok))
    for x in st["records"]:
        assert x["n_tokens"] == len(x["completion_ids"]) <= 6
        assert (x["finish"] == "stop") == (x["completion_ids"][-1] in stops)
    run = json.load(open(res / "drift_run.json"))
    assert [m["tag"] for m in run["missing"]] == ["gone"] and set(run["done"]) == {"self", "other"}
    r2 = subprocess.run([sys.executable, "-m", "bc2rl", "drift", "--config", str(tmp / "drift.yaml")], cwd=ROOT,
                        capture_output=True, text=True)
    assert r2.returncode == 0 and "done=2" in r2.stdout


def test_r1_drift_generated_configs():
    """The real generated stage-R1 configs (configs/r/drift-s*.json; protocol 14.1)."""
    import glob
    paths = sorted(glob.glob(os.path.join(ROOT, "configs", "r", "drift-s*.json")))
    assert len(paths) == 6
    sha = json.load(open(os.path.join(ROOT, "configs", "a6", "adapter_sha256.json")))
    for p in paths:
        c = json.load(open(p))
        d = c["drift"]
        s = int(p.rsplit("-s", 1)[1].split(".")[0])
        assert d["select"] == {"yd3": 192, "yd5": 64} and d["n_ref"] == 16 and d["cap"] == 256 and d["data_name"] == "a6"
        refs = {r["tag"]: r for r in d["refs"]}
        assert set(refs) == {f"b{k}-s{s}" for k in (16, 32, 64, 128)}
        assert refs[f"b32-s{s}"]["adapter"] == f"runs/bc2rl-y4-bc-path-s{s}/adapters/step0032.safetensors"
        assert len(d["score"]) == 30 and len({x["tag"] for x in d["score"]}) == 30
        for x in d["score"]:
            step = int(x["tag"].split("-")[0][1:])
            assert x["ref"] == f"b{step}-s{s}" and x["tag"].endswith(("-r064", "-r128", "-r256"))
            assert x["adapter"].endswith(f"round{int(x['tag'][-3:]):04d}.safetensors")
            if x["tag"].endswith("-r256") and x["adapter"] in sha:
                assert x.get("sha256") in (None, sha[x["adapter"]])
        assert all(r["adapter"] in sha or "y4-bc-path" in r["adapter"] for r in d["refs"])
        if PD and os.path.isdir(os.path.join(PD, "runs")):
            fin = [x for x in d["score"] if x["tag"].endswith("-r256")]
            assert all(os.path.exists(os.path.join(PD, x["adapter"])) for x in fin), "a final adapter is missing"
