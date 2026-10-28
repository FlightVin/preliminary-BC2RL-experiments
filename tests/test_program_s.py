"""Tests for A6 program S stage S3 online options (protocol Section 13.4) on the tiny random Qwen2 of test_a6.py (CPU):
class pool from ext panels, per-class streams, class masking, per-class signal counts, class-gradient diagnostic."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_a6 import env, needs_tok, online_cfg, run_cli, write  # noqa: E402,F401

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
