"""Tests for amendment A6 (protocol/AMENDMENT-A6-2026-10-26.md): analysis estimators on synthetic data, the matching
rule, problem-structure helpers, the real A6 configs against the frozen ones, the worker's `requires` /
stop-claiming options, and (tiny random Qwen2, CPU) the MASK / T3 online options and the xeval cap extension."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import yaml

from bc2rl import task
from bc2rl import yanalysis as Y
from bc2rl.extension import CODE, lenient_success
from bc2rl.xanalysis import boot_idx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PD = os.environ.get("PROJ_DATA", "")
SEEDS = (11, 22, 33, 44, 55, 66)


# ----------------------------------------------------------------------------- synthetic panels

def synth(rng, n=768, arms=("M",), change=None, code_mix=None):
    """Initial p: step 16 ~ Beta(0.5, 3), step 128 ~ Beta(0.9, 1.4). change(step, arm, p) -> final p.
    Every attempt gets a failure code; code_mix(step, arm) -> probabilities over FAIL_CODES for failures."""
    counts, pids = {}, [f"y-{j:05d}" for j in range(n)]
    fails = np.array(list(Y.FAIL_CODES))

    def codes(succ, step, arm):
        mix = np.ones(len(fails)) / len(fails) if code_mix is None else np.asarray(code_mix(step, arm))
        out = np.where(succ, "S", rng.choice(fails, size=succ.shape, p=mix))
        return ["".join(r) for r in out]

    for s in SEEDS:
        for step, (a, b) in ((16, (0.5, 3.0)), (128, (0.9, 1.4))):
            p = rng.beta(a, b, n)
            att = rng.random((n, 48)) < p[:, None]
            cs = codes(att, step, "init")
            counts[f"b{step}-s{s}"] = {q: {"n": 48, "c": int(x.sum()), "c_a": int(x[:24].sum()),
                                          "c_b": int(x[24:].sum()), "c_x": int(x.sum()), "c_a_x": int(x[:24].sum()),
                                          "c_b_x": int(x[24:].sum()), "codes": c, "x_codes": c}
                                      for q, x, c in zip(pids, att, cs)}
            for arm in arms:
                for m in ("grpo", "filtered"):
                    pf = p if change is None else change(step, arm, p)
                    fa = rng.random((n, 32)) < pf[:, None]
                    cf = codes(fa, step, arm)
                    counts[f"b{step}-{m}-{arm}-s{s}"] = {q: {"n": 32, "c": int(x.sum()), "c_x": int(x.sum()),
                                                             "codes": c, "x_codes": c}
                                                         for q, x, c in zip(pids, fa, cf)}
    return Y.Evals(counts)


def pairs(arm="M", m="grpo"):
    return {(m, s): (f"b16-s{s}", f"b16-{m}-{arm}-s{s}", f"b128-s{s}", f"b128-{m}-{arm}-s{s}") for s in SEEDS}


def test_decomposition_sums_to_delta_k1():
    ev = synth(np.random.default_rng(1), n=300, code_mix=lambda st, arm: [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
               if st == 128 and arm != "init" else [0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
    w = Y.pooled_weights(ev, [f"b{n}-s{s}" for n in (16, 128) for s in SEEDS])
    tot = Y.init_contrast(ev, pairs(), w, None, lambda k: k[1], K=1)[0]["point"]
    parts = sum(Y.init_contrast(ev, pairs(), w, None, lambda k: k[1], K=1, code=c)[0]["point"] for c in Y.FAIL_CODES)
    assert abs(tot - parts) < 1e-12
    trunc = Y.init_contrast(ev, pairs(), w, None, lambda k: k[1], K=1, code="T")[0]["point"]
    assert trunc > 0                                  # planted: extended finals fail more often by truncation


def test_y3_contrasts_detect_planted_cross_difficulty_effect():
    def change(step, arm, p):                         # extended loses solutions only when harder updates exist
        return np.where((step == 128) & (arm == "M") & (p < 0.5), p * 0.4, p)
    ev = synth(np.random.default_rng(2), arms=("M", "MASK"), change=change)
    w = Y.pooled_weights(ev, [f"b{n}-s{s}" for n in (16, 128) for s in SEEDS])
    idx = boot_idx(len(ev.pids), 300, 0)
    _, PM, RM = Y.init_contrast(ev, pairs("M"), w, idx, lambda k: k[1])
    _, PK, RK = Y.init_contrast(ev, pairs("MASK"), w, idx, lambda k: k[1])
    c1 = Y.diff_of(PM, RM, PK, RK, lambda k: k[1], idx)
    c2 = Y.group_units(PK, RK, lambda k: k[1], idx)
    assert c1["sign"] == "+" and c1["both_exclude_0"]
    assert c2["sign"] == "0"


def test_null_has_no_effect_and_intervals_cover():
    ev = synth(np.random.default_rng(3), arms=("M",))
    w = Y.pooled_weights(ev, [f"b{n}-s{s}" for n in (16, 128) for s in SEEDS])
    idx = boot_idx(len(ev.pids), 300, 0)
    summ = Y.init_contrast(ev, pairs(), w, idx, lambda k: k[1])[0]
    assert abs(summ["point"]) < 0.05 and summ["boot95"][0] < 0 < summ["boot95"][1]


def test_summarize_contrast_requires_both_intervals():
    reps = {s: np.full(100, 0.1) + np.linspace(-0.01, 0.01, 100) for s in SEEDS}
    pts = {s: v for s, v in zip(SEEDS, (0.3, -0.2, 0.25, -0.1, 0.2, 0.0))}   # seed spread: t95 includes 0
    out = Y.summarize_contrast(pts, reps)
    assert out["boot95"][0] > 0 and out["t95"][0] < 0 and not out["both_exclude_0"] and out["sign"] == "0"


def test_npmle_recovers_two_point_mixture():
    rng = np.random.default_rng(4)
    p = np.where(rng.random(3000) < 0.5, 0.1, 0.6)
    p_grid, pi = Y.npmle(rng.binomial(48, p), 48, iters=1500)
    assert abs((p_grid * pi).sum() - p.mean()) < 0.01
    post = Y.posterior_mean_by_bin(p_grid, pi, 24)
    assert post[0] < post[1] < post[2]


def test_match_rule():
    a = {"pass1_3": 0.30, "pass1_5": 0.05, "hist": [0.2, 0.2, 0.2, 0.2, 0.1, 0.1], "bin_base": [0.08, 0.17, 0.33]}
    b = dict(a, pass1_3=0.315, hist=[0.18, 0.22, 0.2, 0.2, 0.1, 0.1], bin_base=[0.09, 0.18, 0.35])
    assert Y.match(a, b)[0]
    assert not Y.match(a, dict(b, pass1_5=0.08))[0]
    assert not Y.match(a, dict(b, bin_base=[0.09, 0.18, 0.40]))[0]
    assert not Y.match(a, dict(b, bin_base=[None, None, None]))[0]


def test_solution_forms_and_text_forms():
    n, forms = Y.solution_forms([2, 3, 4], 10)
    assert ("*", "+", "L") in forms and ("*", "-", "L") in forms and n >= 2
    assert Y.form_of_text("2 * 3 = 6\n6 + 4 = 10") == ("*", "+", "L")
    assert Y.form_of_text("2 + 3 = 5\n20 - 5 = 15") == ("+", "-", "R")
    assert Y.form_of_text("2 + 3 = 5") is None


def test_lenient_parse_and_codes():
    text = "We compute 2*3=6, then 6+4=10."
    assert task.verify([2, 3, 4], 10, text)["reward"] == 0
    assert lenient_success([2, 3, 4], 10, text) == 1
    assert lenient_success([2, 3, 4], 10, "2 + 3 = 5\n5 + 4 = 9") == 0
    assert set(CODE.values()) == set(Y.CODES)


def test_a6_configs_match_frozen_protocol():
    """Every A6 config differs from its frozen source only in the keys the protocol names."""
    wl = json.load(open(os.path.join(ROOT, "configs/worklists/a6.json")))
    assert wl["name"] == "a6" and len(wl["tasks"]) == 73
    assert len({t["id"] for t in wl["tasks"]}) == 73
    for s in SEEDS:
        b = json.load(open(os.path.join(ROOT, f"configs/a6/bc-path-s{s}.json")))
        b0 = json.load(open(os.path.join(ROOT, b["a6_source"])))
        assert {k: v for k, v in b.items() if k not in ("run_id", "bc", "a6_source")} == \
            {k: v for k, v in b0.items() if k not in ("run_id", "bc")}
        assert {k: v for k, v in b["bc"].items() if k not in ("steps", "save_steps")} == \
            {k: v for k, v in b0["bc"].items() if k not in ("steps", "save_steps")}
        assert b["bc"]["steps"] == 128 and b["bc"]["save_steps"] == [16, 32, 64, 128]
        for i in ("light", "extended"):
            for arm, key in (("mask", "train_inputs"), ("t3", "pool_inputs")):
                o = json.load(open(os.path.join(ROOT, f"configs/a6/on-{i}-grpo-{arm}-s{s}.json")))
                o0 = json.load(open(os.path.join(ROOT, o["a6_source"])))
                strip = lambda c: {**{k: v for k, v in c.items() if k not in ("run_id", "online", "a6_source")},
                                   "online": {k: v for k, v in c["online"].items()
                                              if k not in ("train_inputs", "pool_inputs", "save_rounds", "init")}}
                assert strip(o) == strip(o0)
                assert o["online"][key] == [3] and o["online"]["init"]["tag"] == o0["online"]["init"]["tag"]
                assert o["online"]["init"]["adapter"] == o0["online"]["init"]["adapter"]
                assert len(o["online"]["init"]["sha256"]) == 64
                extra = set(o["online"]["save_rounds"]) - set(o0["online"]["save_rounds"])
                assert extra == ({85} if arm == "t3" else set())
        for n in (32, 64):
            o = json.load(open(os.path.join(ROOT, f"configs/a6/on-step{n}-grpo-s{s}.json")))
            o0 = json.load(open(os.path.join(ROOT, o["a6_source"])))
            keep = ("method", "lr", "rounds", "prompts_per_round", "group", "cap", "micro", "clip", "prompt_split",
                    "stream_seed")
            assert {k: o["online"][k] for k in keep} == {k: o0["online"][k] for k in keep}
            assert o["online"]["init"]["adapter"].endswith(f"step{n:04d}.safetensors")
    ev = json.load(open(os.path.join(ROOT, "configs/a6/xeval-yf-init-s11.json")))["xeval"]
    assert ev["cap"] == 256 and ev["cap_ext"] == 1024 and ev["emit_counts"] is False and ev["data_name"] == "a6"
    assert [c["attempts"] for c in ev["checkpoints"]] == [48, 48, 48] and all(c["split"] == 24 for c in ev["checkpoints"])
    d = json.load(open(os.path.join(ROOT, "configs/a6/ydata.json")))["xdata"]
    assert d["exclude_ext"] == ["x5"] and d["panels"]["yf3"]["count"] == 768


def test_worker_requires_and_stop_claiming(tmp_path):
    pd = tmp_path / "pd"
    pd.mkdir()
    wl = {"name": "wtest", "tasks": [{"id": "needs", "requires": ["missing/file"], "steps": []},
                                     {"id": "free", "steps": []}]}
    wp = tmp_path / "wl.json"
    wp.write_text(json.dumps(wl))
    env = dict(os.environ, PROJ_DATA=str(pd), SLURM_JOB_ID="77")
    r = subprocess.run([sys.executable, "scripts/worker.py", "--worklist", str(wp), "--results-dir",
                        str(tmp_path / "r1"), "--requires-wait-s", "0"], cwd=ROOT, env=env, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    st = json.load(open(tmp_path / "r1" / "worker_status.json"))
    assert st["tasks"]["free"] == "done" and st["tasks"]["needs"] == "requires-missing"
    assert not (pd / "claims" / "wtest" / "needs").exists()
    wl["name"] = "wtest2"
    wp.write_text(json.dumps(wl))
    r = subprocess.run([sys.executable, "scripts/worker.py", "--worklist", str(wp), "--results-dir",
                        str(tmp_path / "r2"), "--stop-claiming-after-s", "0", "--requires-wait-s", "0"],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    st = json.load(open(tmp_path / "r2" / "worker_status.json"))
    assert st["tasks"]["free"] == "left-for-next-worker" and st.get("stopped_claiming")
    assert not (pd / "claims" / "wtest2" / "free").exists()


# ----------------------------------------------------------------------------- tiny model (CPU)

needs_tok = pytest.mark.skipif(not os.path.exists(os.path.join(PD, "model_revision.json")),
                               reason="needs PROJ_DATA with the downloaded tokenizer")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    tmp = tmp_path_factory.mktemp("a6")
    pd = tmp / "pd"
    (pd / "data" / "t1").mkdir(parents=True)
    info = json.load(open(os.path.join(PD, "model_revision.json")))
    json.dump(info, open(pd / "model_revision.json", "w"))
    torch.manual_seed(0)
    tiny = Qwen2ForCausalLM(Qwen2Config(vocab_size=151936, hidden_size=32, intermediate_size=64,
                                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                        max_position_embeddings=4096, tie_word_embeddings=True))
    tiny.save_pretrained(tmp / "tiny")
    cfg = {"run_id": "x", "proj_data": str(pd), "device": "cpu", "test_mode": True, "seed": 11,
           "prompt_variant": "format", "model_path_override": str(tmp / "tiny"),
           "model": {"lora": {"r": 4, "alpha": 8, "dropout": 0.0}},
           "data": {"version": "t1", "main_seed": 1, "dev_seed": 2, "min_value": 2, "max_value": 20,
                    "max_target": 1000, "max_abs_intermediate": 10000, "ops": ["+", "-", "*"],
                    "horizons": [3, 5, 10],
                    "sizes": {"dev": 3, "devdemo": 6, "bc": 12, "online": 12, "val": 3, "test": 3}},
           "val": {"split": "val", "attempts": 1, "cap": 4}}
    run_cli("generate", write(tmp, "gen", cfg))
    return tmp, cfg


def run_cli(cmd, cfg_path, expect=0):
    r = subprocess.run([sys.executable, "-m", "bc2rl", cmd, "--config", str(cfg_path)], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == expect, r.stdout[-3000:] + r.stderr[-3000:]
    return r


def write(tmp, name, cfg):
    p = tmp / f"{name}.yaml"
    yaml.safe_dump(cfg, open(p, "w"))
    return p


def online_cfg(base, rid, **extra):
    on = {"method": "grpo", "lr": 1e-3, "rounds": 3, "prompts_per_round": 6, "group": 2, "cap": 6, "micro": 4,
          "clip": 0.2, "prompt_split": "online", "debug_reward": "random", "init": {"adapter": "none", "tag": "nobc"},
          "save_rounds": [3], "val_rounds": []}
    on.update(extra)
    return dict(base, run_id=rid, online=on)


@needs_tok
def test_mask_with_all_inputs_is_bit_identical_and_mask_removes_terms(env):
    tmp, base = env
    from bc2rl.modeling import file_sha256
    for rid, extra in (("m-def", {}), ("m-all", {"train_inputs": [3, 5, 10]}), ("m-3", {"train_inputs": [3]}),
                       ("t-3", {"pool_inputs": [3]})):
        run_cli("online", write(tmp, rid, online_cfg(base, rid, **extra)))
    rd = lambda rid: os.path.join(base["proj_data"], "runs", rid)
    sha = {rid: file_sha256(os.path.join(rd(rid), "adapters", "round0003.safetensors"))
           for rid in ("m-def", "m-all", "m-3")}
    assert sha["m-def"] == sha["m-all"]
    rows = [json.loads(l) for l in open(os.path.join(rd("m-3"), "metrics.jsonl"))]
    assert all(r["used_by_inputs"]["5"] == 0 and r["used_by_inputs"]["10"] == 0 for r in rows)
    rows_def = [json.loads(l) for l in open(os.path.join(rd("m-def"), "metrics.jsonl"))]
    assert sum(r["used_by_inputs"]["5"] + r["used_by_inputs"]["10"] for r in rows_def) > 0
    assert rows[0]["successes"] == rows_def[0]["successes"]      # same first-round sampling (same stream and seeds)
    from bc2rl.runtime import read_jsonl_gz
    recs = read_jsonl_gz(os.path.join(rd("t-3"), "records", "round0001.jsonl.gz"))
    assert recs and all("-3-" in r["pid"] for r in recs)


@needs_tok
def test_xeval_cap_extension(env):
    tmp, base = env
    from bc2rl.data import data_dir
    from bc2rl.extension import draw_fresh
    from bc2rl.modeling import load_policy, save_adapter
    from bc2rl.runtime import read_jsonl_gz
    d = os.path.join(data_dir(base), "ext", "a6t")
    os.makedirs(d)
    out, _ = draw_fresh(9, task.GenConfig(), {"p3": (3, 3)}, set())
    task.save_split(os.path.join(d, "p3.jsonl"), out["p3"])
    json.dump({"panels": {"p3": {"count": 3, "n_inputs": 3, "sha256": task.manifest_hash(out["p3"])}}},
              open(os.path.join(d, "manifest.json"), "w"))
    model = load_policy(base["model_path_override"], base["model"]["lora"], seed=0, device="cpu")
    ad = os.path.join(base["proj_data"], "ad.safetensors")
    save_adapter(model, ad)
    xe = {"data_name": "a6t", "panels": ["p3"], "variant": "format", "cap": 4, "cap_ext": 12, "seed": 6,
          "batch_seqs": 8, "ext_batch_seqs": 5, "emit_counts": False,
          "checkpoints": [{"tag": "b16-s11", "adapter": ad, "attempts": 4, "split": 2}]}
    cfg = dict(base, run_id="xe", xeval=xe)
    res = tmp / "res"
    r = subprocess.run([sys.executable, "-m", "bc2rl", "xeval", "--config", str(write(tmp, "xe", cfg)),
                        "--results-dir", str(res)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    ed = os.path.join(base["proj_data"], "evals", "ext", "a6t")
    s = json.load(open(os.path.join(ed, "b16-s11.p3.summary.json")))
    recs = read_jsonl_gz(os.path.join(ed, "b16-s11.p3.jsonl.gz"))
    assert len(recs) == 12 and s["protocol"]["cap_ext"] == 12
    for x in recs:
        assert "_completion_ids" not in x and x["code"] == CODE[x["error"]]
        if x["finish"] == "length":
            assert x["n_tokens"] == 4 and 4 <= x["x_n_tokens"] <= 12 and "x_text" in x
            assert x["x_finish"] != "length" or x["x_n_tokens"] == 12
        else:
            assert x["x_reward"] == x["reward"] and x["x_code"] == x["code"]
    assert s["n_continued"] == sum(x["finish"] == "length" for x in recs) > 0
    for row in s["counts"]:
        assert len(row["codes"]) == len(row["x_codes"]) == 4 and row["c_a_x"] + row["c_b_x"] == row["c_x"]
    assert not (res / "xcounts.csv").exists() and (res / "xeval_b16-s11.p3.json").exists()
    r2 = subprocess.run([sys.executable, "-m", "bc2rl", "xeval", "--config", str(write(tmp, "xe", cfg))],
                        cwd=ROOT, capture_output=True, text=True)
    assert "reuse" in r2.stdout


# ----------------------------------------------------------------------------- look 1 end to end (synthetic)

def _look1_fixture(tmp, n=200):
    """Synthetic yf panels, evaluations, matching file and runs for `scripts/y_look1.py`. Planted effect: arm M from
    step 128 (and 64, 32 in proportion) loses more than from step 16; MASK / T3 / T3E lose the same at both steps."""
    import torch
    from safetensors.torch import save_file
    from bc2rl.extension import draw_fresh
    from bc2rl.runtime import write_jsonl_gz
    rng = np.random.default_rng(3)
    pd = tmp / "pd"
    d = pd / "data" / "t1" / "ext" / "a6"
    d.mkdir(parents=True)
    out, _ = draw_fresh(5, task.GenConfig(), {"yf3": (3, n), "yf5": (5, 64)}, set())
    man = {"panels": {}}
    for k, probs in out.items():
        task.save_split(str(d / f"{k}.jsonl"), probs)
        man["panels"][k] = {"count": len(probs), "n_inputs": probs[0].n_inputs, "sha256": task.manifest_hash(probs)}
    json.dump(man, open(d / "manifest.json", "w"))
    cfg = tmp / "ydata.json"
    json.dump({"proj_data": str(pd), "data": {"version": "t1"}}, open(cfg, "w"))
    ev_dir = tmp / "evals"
    ev_dir.mkdir()
    drop = {16: 0.85, 32: 0.75, 64: 0.65, 128: 0.45}
    beta = {16: (0.5, 3.0), 32: (0.6, 2.4), 64: (0.75, 1.9), 128: (0.9, 1.4)}

    def emit(tag, panel, probs, p, n_att, split, records):
        att = rng.random((len(probs), n_att)) < p[:, None]
        fails = np.array(list(Y.FAIL_CODES))
        cod = np.where(att, "S", rng.choice(fails, size=att.shape))
        rows, recs = [], []
        for q, x, c in zip(probs, att, cod):
            c = "".join(c)
            r = {"pid": q.pid, "n": n_att, "c": int(x.sum()), "c_x": int(x.sum()), "codes": c, "x_codes": c}
            if split:
                r.update(c_a=int(x[:split].sum()), c_b=int(x[split:].sum()), c_a_x=int(x[:split].sum()),
                         c_b_x=int(x[split:].sum()))
            rows.append(r)
            if records:
                recs += [{"pid": q.pid, "attempt": j, "reward": int(x[j]),
                          "text": task.completion_text(q) if x[j] else "no"} for j in range(n_att)]
        json.dump({"tag": tag, "panel": panel, "counts": rows}, open(ev_dir / f"{tag}.{panel}.summary.json", "w"))
        if records:
            write_jsonl_gz(str(ev_dir / f"{tag}.{panel}.jsonl.gz"), recs)

    for s in SEEDS:
        for step in STEPS_ALL:
            p3 = rng.beta(*beta[step], len(out["yf3"]))
            p5 = rng.beta(0.3, 6.0, len(out["yf5"]))
            rec = step in (16, 128)
            emit(f"b{step}-s{s}", "yf3", out["yf3"], p3, 48, 24, rec)
            emit(f"b{step}-s{s}", "yf5", out["yf5"], p5, 48, 24, False)
            arms = [("grpo", "M")] + ([("filtered", "M"), ("grpo", "MASK"), ("grpo", "T3"), ("grpo", "T3E")]
                                      if rec else [])
            for m, arm in arms:
                f = drop[step] if arm == "M" else 0.85
                emit(f"b{step}-{m}-{arm}-s{s}", "yf3", out["yf3"], p3 * f, 32, 0, rec and arm == "M")
                emit(f"b{step}-{m}-{arm}-s{s}", "yf5", out["yf5"], p5 * f, 32, 0, False)
    prof = {f"b{n_}-s{s}": {"pass1_3": float(rng.random())} for n_ in STEPS_ALL for s in SEEDS}
    mt = {"rule": "test", "profiles": prof, "matched_pairs": [{"low": "b16-s44", "high": "b32-s11"},
                                                             {"low": "b64-s33", "high": "b128-s66"}]}
    json.dump(mt, open(tmp / "match.json", "w"))
    runs = tmp / "runs"
    sd = lambda v: {"a": torch.full((4, 2), float(v)), "b": torch.zeros(3)}
    for s in SEEDS:
        for n_ in (16, 128):
            p = runs / f"bc2rl-s1-bc-broad-s{s}" / "adapters"
            p.mkdir(parents=True, exist_ok=True)
            save_file(sd(0), str(p / f"step{n_:04d}.safetensors"))
    p = runs / "bc2rl-y3-light-grpo-mask-s11"
    (p / "adapters").mkdir(parents=True)
    save_file(sd(1), str(p / "adapters" / "round0064.safetensors"))
    with open(p / "metrics.jsonl", "w") as f:
        f.write(json.dumps({"used_by_inputs": {"3": 5, "5": 0, "10": 0}}) + "\n")
    return cfg, ev_dir, tmp / "match.json", runs


STEPS_ALL = (16, 32, 64, 128)


def test_look1_end_to_end_synthetic(tmp_path):
    cfg, ev_dir, match, runs = _look1_fixture(tmp_path)
    out = tmp_path / "out"
    cmd = [sys.executable, "scripts/y_look1.py", "--evals", str(ev_dir), "--matching", str(match), "--runs", str(runs),
           "--config", str(cfg), "--out", str(out), "--n-boot", "300"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    res = json.load(open(out / "look1.json"))
    man = json.load(open(out / "look1_manip.json"))
    y3 = res["Y3"]["cap256_K16"]
    assert y3["Delta_M"]["point"] > 0.05 and y3["C1_DeltaM_minus_DeltaMASK"]["sign"] == "+"
    assert abs(y3["C2_Delta_MASK"]["point"]) < 0.05 and y3["C2_Delta_MASK"]["sign"] != "+"
    assert res["Y2a_class"] == "not explained by truncation"           # c_x == c in the fixture
    assert res["Y2a_delta_256_minus_1024"]["point"] == 0
    assert set(res["Y2b_cap256"]["codes"]) == set(Y.FAIL_CODES)
    assert res["Y4"]["shape"] in ("gradual", "threshold", "non-monotonic", "undetermined")
    assert res["Y4"]["E_step128"]["point"] > res["Y4"]["E_step16"]["point"]
    assert set(res["Y4"]["matched"]["pairs"]) == {"b16-s44|b32-s11", "b64-s33|b128-s66"}
    assert {"H-W1_pass16", "H-W1_pass1", "H-F1_E_I_step16", "H-S1_step128", "H-S2", "H-S3_step16"} <= set(res["S10_1"])
    assert res["S10_1"]["n_needs_mul"] > 0
    assert man["used_by_inputs"] == {"step16-mask-s11": {"3": 5, "5": 0, "10": 0}}
    assert man["displacement_norms"]["step16-MASK-s11-r64"] == pytest.approx(8 ** 0.5)
    assert len(res["whole_panel"]) == 2 * (24 + 72)
    os.remove(ev_dir / "b64-grpo-M-s66.yf5.summary.json")             # an incomplete set is refused
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "look 1 refused" in r.stderr
