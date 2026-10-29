"""Tests for A6 stage R3 (protocol 14.3, 14.9): pool screening, initial-policy caches and the four-arm block runner with
the owner's supervised loss L = -(1/(P*cap)) sum_i m_i sum_t log p(y_it), on the tiny random Qwen2 of test_a6.py (CPU)."""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_a6 import ROOT, env, needs_tok, write  # noqa: E402,F401
from test_program_s import _ext  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from bc2rl.rsource import sup_loss_sum  # noqa: E402

CAP, P = 40, 4


def test_sup_loss_hand_batch():
    """Two accepted rows of 3 and 1 tokens (incl. the stop token), padding ignored, fixed denominator P x cap."""
    lp = torch.tensor([[-0.5, -1.0, -0.25, -9.0], [-2.0, -9.0, -9.0, -9.0]])
    lens = torch.tensor([3, 1])
    m = (torch.arange(4).unsqueeze(0) < lens.unsqueeze(1)).float()
    assert float(sup_loss_sum(lp, m)) == pytest.approx(3.75)
    assert float(sup_loss_sum(lp, m) / (32 * 256)) == pytest.approx(3.75 / 8192)
    assert float(sup_loss_sum(lp[:0], m[:0])) == 0.0


def _run(args, expect=0):
    r = subprocess.run([sys.executable, "-m", "bc2rl", *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == expect, r.stdout[-3000:] + r.stderr[-3000:]
    return r


@pytest.fixture(scope="module")
def r3(env):
    from bc2rl.modeling import load_policy, save_adapter
    tmp, base = env
    _ext(base, "rs1t", {"rsc": (3, 40)})
    model = load_policy(base["model_path_override"], base["model"]["lora"], seed=0, device="cpu")
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad and "lora_B" in n:
                p.add_(0.05 * torch.randn_like(p))
    ad = os.path.join(base["proj_data"], "r3-init.safetensors")
    save_adapter(model, ad)
    ck = {"tag": "c0", "adapter": ad, "role": "conf"}
    prep = dict(base, run_id="rp", seed=7, _debug_verify="random", rprep={
        "out": "t", "data_name": "rs1t", "panel": "rsc", "cap": 8, "batch_seqs": 64,
        "screen": {"attempts": 2, "seed": 10, "chunk": 16, "target": 6, "max": 40, "fallback": 3},
        "cache": {"attempts": 8, "batch": 4, "seed": 11}, "screen_checkpoints": [ck, dict(ck, tag="c1")],
        "cache_checkpoints": [ck, dict(ck, tag="c1")]})
    r = _run(["rprep", "--config", str(write(tmp, "rprep", prep)), "--results-dir", str(tmp / "rpres")])
    return tmp, base, ad, r


@needs_tok
def test_rprep_pool_screening_and_caches(r3):
    tmp, base, ad, r = r3
    out = os.path.join(base["proj_data"], "rsource", "t")
    pool = json.load(open(os.path.join(out, "pool.json")))
    scr = json.load(open(os.path.join(out, "screen.json")))
    assert pool["rule"] == "target" and len(pool["pool"]) == 6 and pool["screened"] == sum(scr["chunks"])
    assert scr["chunks"][0] == 16
    from bc2rl.extension import load_ext
    ext, _ = load_ext(base, "rs1t")
    order = [p.pid for p in ext["rsc"]]
    q = [x for x in order[:pool["screened"]] if all(scr["counts"][t][x] > 0 for t in ("c0", "c1"))]
    assert pool["pool"] == q[:6] and pool["qualified"] == len(q)
    assert all(v["attempts"] == 2 * n for v, n in zip([scr["cost"][f"c0@{s}"] for s in (0, 16)][:len(scr["chunks"])],
                                                   scr["chunks"]))
    c0 = json.load(open(os.path.join(out, "caches", "c0.json")))
    assert set(c0["targets"]) == set(pool["pool"])
    found = [t for t in c0["targets"].values() if t]
    assert all(0 <= t["attempt"] < 8 for t in found) and c0["summary"]["found"] == len(found)
    res = json.load(open(tmp / "rpres" / "rprep_summary.json"))
    assert res["common_pool"] == len(json.load(open(os.path.join(out, "common_pool.json")))["pool"])


@needs_tok
def test_rprep_gate_fails_below_fallback(r3):
    tmp, base, ad, _ = r3
    cfg = dict(base, run_id="rp2", seed=7, _debug_verify="random", rprep={
        "out": "t2", "data_name": "rs1t", "panel": "rsc", "cap": 8, "batch_seqs": 64,
        "screen": {"attempts": 1, "seed": 10, "chunk": 8, "target": 30, "max": 8, "fallback": 30},
        "cache": {"attempts": 4, "batch": 4, "seed": 11}, "screen_checkpoints": [{"tag": "c0", "adapter": ad}],
        "cache_checkpoints": [{"tag": "c0", "adapter": ad}]})
    _run(["rprep", "--config", str(write(tmp, "rprep2", cfg))], expect=3)
    assert json.load(open(os.path.join(base["proj_data"], "rsource", "t2", "pool.json")))["rule"] == "failed"


def _fix_caches(base, tok):
    """The tiny random model rarely ends with a stop token: give every I target a valid stop-terminated id list."""
    out = os.path.join(base["proj_data"], "rsource", "t")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    pool = json.load(open(os.path.join(out, "common_pool.json")))["pool"]
    assert len(pool) >= 4
    cp = os.path.join(out, "caches", "c0.json")
    c = json.load(open(cp))
    for j, pid in enumerate(pool):
        c["targets"][pid] = {"ids": [100 + j, 200 + j, im_end], "text": "x", "attempt": 0, "finish": "stop"}
    json.dump(c, open(cp, "w"))


def _block(base, ad, rid, **extra):
    from bc2rl.modeling import file_sha256
    cfg = dict(base, run_id=rid, seed=7, _debug_verify="all", rsource={
        "prompts_per_round": P, "rounds": 3, "cap": CAP, "lr": 1e-3, "micro": 3, "batch_seqs": 64,
        "fresh": {"attempts": 4, "batch": 2}, "save_rounds": [0, 1, 2, 3], "mask_limit": 0.05,
        "data_name": "rs1t", "panel": "rsc", "prep_out": "t", "cache_tag": "c0",
        "init": {"tag": "c0", "adapter": ad, "sha256": file_sha256(ad)}, "stream_seed": 7, "sample_seed": 7})
    cfg.update(extra)
    return cfg


@needs_tok
def test_rsource_loss_mask_empty_round_and_adamw(r3):
    from safetensors.torch import load_file
    from bc2rl import task
    from bc2rl.data import data_dir
    from bc2rl.extension import load_ext
    from bc2rl.modeling import chat_prompt, load_adapter, load_policy, load_tokenizer, model_info
    from bc2rl.online import prompt_stream
    from bc2rl.runtime import read_jsonl_gz
    tmp, base, ad, _ = r3
    tok = load_tokenizer(model_info(base["proj_data"])["path"])
    _fix_caches(base, tok)
    # round 1: occurrence 0 has no F target, occurrence 1 no O target; round 2: every F fails (empty batch)
    fail = [[1, 0, ["F"]], [1, 1, ["O"]]] + [[2, k, ["F"]] for k in range(P)]
    cfg = _block(base, ad, "rsb", _debug_fail=fail)
    _run(["rsource", "--config", str(write(tmp, "rsb", cfg)), "--results-dir", str(tmp / "rsres")])
    rd = os.path.join(base["proj_data"], "runs", "rsb")
    rows = [json.loads(x) for x in open(os.path.join(rd, "metrics.jsonl"))]
    assert [r["masked"] for r in rows] == [2, P, 0]
    assert rows[1]["arms"] == {} and set(rows[0]["arms"]) == {"S", "I", "F", "O"}
    s = json.load(open(tmp / "rsres" / "summary.json"))
    assert s["counters"]["opt_steps"] == {a: 2 for a in "SIFO"} and s["counters"]["skipped_rounds"] == 1
    assert s["rounds"] == 3 and s["counters"]["masked"] == P + 2 and s["exposure_matching_failed"]
    recs = read_jsonl_gz(os.path.join(rd, "records", "round0001.jsonl.gz"))
    assert [x["accepted"] for x in recs] == [False, False] + [True] * (P - 2)
    ad_path = lambda arm, r: os.path.join(rd, "adapters", arm, f"round{r:04d}.safetensors")
    for arm in "SIFO":            # the empty round changes no arm; round 0 equals the starting adapter
        a1, a2 = load_file(ad_path(arm, 1)), load_file(ad_path(arm, 2))
        assert all(torch.equal(a1[k], a2[k]) for k in a1)
        a0, ai = load_file(ad_path(arm, 0)), load_file(ad)
        assert all(torch.equal(a0[k], ai[k]) for k in ai)
    # hand computation of round 1, arm S: loss from the starting policy, then one clipped AdamW step
    ext, _ = load_ext(base, "rs1t")
    pool_ids = json.load(open(os.path.join(base["proj_data"], "rsource", "t", "common_pool.json")))["pool"]
    by = {p.pid: p for p in ext["rsc"]}
    pool = [by[x] for x in pool_ids]
    occ = prompt_stream(len(pool), P, 3, 7)[0]
    fmt = task.load_split(os.path.join(data_dir(base), "format.jsonl"))[0]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    model = load_policy(base["model_path_override"], base["model"]["lora"], seed=7, device="cpu")
    load_adapter(model, ad)
    params = [p for p in model.parameters() if p.requires_grad]
    total, seqs = 0.0, []
    for k in range(2, P):
        p = pool[occ[k]]
        pi = tok(chat_prompt(tok, task.prompt_text(p, base["prompt_variant"], fmt)), add_special_tokens=False).input_ids
        ci = tok(task.completion_text(p), add_special_tokens=False).input_ids + [im_end]
        ids = torch.tensor([pi + ci])
        lp = torch.log_softmax(model(input_ids=ids).logits[0, :-1].float(), -1)
        tgt = ids[0, 1:]
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[len(pi) - 1:]
        assert tok_lp.numel() == len(ci)                     # every response token, incl. the stop token
        total = total - float(tok_lp.sum())
        seqs.append((pi, ci))
    # independent unbatched forward (BF16 base: batch shape changes the numerics slightly)
    assert rows[0]["arms"]["S"]["loss"] == pytest.approx(total / (P * CAP), rel=2e-2)
    assert rows[0]["arms"]["S"]["tokens"] == sum(len(tok(task.completion_text(pool[occ[k]]),
                                                         add_special_tokens=False).input_ids) + 1 for k in range(2, P))
    # the same micro-batch layout as the runner (micro 3): exact loss, then one clipped AdamW step
    from bc2rl.modeling import gather_completion, seq_logprobs
    model.train()
    ltot = 0.0
    for sl, lp, cm, _ in seq_logprobs(model, seqs, tok.pad_token_id, 3, grad=True):
        lp_g = gather_completion(lp, cm, CAP)
        lens = torch.tensor([len(c) for _, c in seqs[sl]])
        m = (torch.arange(CAP).unsqueeze(0) < lens.unsqueeze(1)).float()
        loss = -(lp_g * m).sum() / (P * CAP)
        loss.backward()
        ltot += float(loss)
    assert rows[0]["arms"]["S"]["loss"] == pytest.approx(ltot, rel=1e-6)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.0)
    opt.step()
    from bc2rl.modeling import adapter_state
    want, got = adapter_state(model), load_file(ad_path("S", 1))
    assert all(torch.allclose(want[k], got[k], atol=1e-7) for k in want)
    i1 = load_file(ad_path("I", 1))
    assert any(not torch.equal(i1[k], got[k]) for k in got)      # separate arms, separate targets


@needs_tok
def test_rsource_resume_is_identical(r3):
    from safetensors.torch import load_file
    from bc2rl.modeling import load_tokenizer, model_info
    tmp, base, ad, _ = r3
    _fix_caches(base, load_tokenizer(model_info(base["proj_data"])["path"]))
    full = _block(base, ad, "rsf")
    _run(["rsource", "--config", str(write(tmp, "rsf", full))])
    cut = _block(base, ad, "rsk", _debug_kill_after_round=2, _debug_ckpt_every=1)
    _run(["rsource", "--config", str(write(tmp, "rsk", cut))], expect=17)
    cut.pop("_debug_kill_after_round")
    r = _run(["rsource", "--config", str(write(tmp, "rsk", cut))])
    assert "resumed rsk at round 2" in r.stdout
    rd = lambda rid, arm: os.path.join(base["proj_data"], "runs", rid, "adapters", arm, "round0003.safetensors")
    for arm in "SIFO":
        a, b = load_file(rd("rsf", arm)), load_file(rd("rsk", arm))
        assert all(torch.equal(a[k], b[k]) for k in a)
    sf = json.load(open(os.path.join(base["proj_data"], "runs", "rsf", "summary.json")))
    sk = json.load(open(os.path.join(base["proj_data"], "runs", "rsk", "summary.json")))
    assert sf["counters"] == sk["counters"]


def test_r3_generated_configs():
    """The real generated stage-R3 configs (configs/r; protocol 14.3, 14.9.1-14.9.2)."""
    from test_a6 import PD
    rd = lambda n: json.load(open(os.path.join(ROOT, "configs", "r", n)))
    sha = json.load(open(os.path.join(ROOT, "configs", "a6", "adapter_sha256.json")))
    x = rd("xdata-rs1.json")["xdata"]
    assert x["name"] == "rs1" and x["seed"] == 20261030 and x["exclude_all_ext"] is True
    assert {k: (v["n_inputs"], v["count"]) for k, v in x["panels"].items()} == {
        "rsc": (3, 2048), "rd3": (3, 256), "rf3": (3, 768), "rf5": (5, 128)}
    rp = rd("rprep.json")["rprep"]
    assert rp["screen"] == {"attempts": 16, "seed": 10, "chunk": 256, "target": 256, "max": 2048, "fallback": 128}
    assert rp["cache"] == {"attempts": 64, "batch": 8, "seed": 11} and rp["cap"] == 256 and rp["panel"] == "rsc"
    conf = rp["screen_checkpoints"]
    assert sorted(c["tag"] for c in conf) == sorted(f"b{st}-s{s}" for s in (11, 22, 33) for st in (16, 128))
    assert all(c["sha256"] == sha[c["adapter"]] and c["role"] == "conf" for c in conf)
    assert rp["cache_checkpoints"][:6] == conf and [c["role"] for c in rp["cache_checkpoints"][6:]] == ["dev", "dev"]
    wl = json.load(open(os.path.join(ROOT, "configs", "worklists", "r3-dev.json")))
    assert [t["id"] for t in wl["tasks"]] == ["r3-prep", "r3-dev-b16-s7", "r3-dev-b128-s7"]
    for init, step in (("light", 16), ("extended", 128)):
        c = rd(f"rs-dev-{init}-s7.json")
        b = c["rsource"]
        assert (b["prompts_per_round"], b["rounds"], b["cap"], b["lr"]) == (32, 256, 256, 1e-5)
        assert b["fresh"] == {"attempts": 64, "batch": 8} and b["mask_limit"] == 0.05
        assert b["save_rounds"] == [0, 32, 64, 128, 192, 256] and b["cache_tag"] == f"dev-b{step}-s7"
        assert c["model"]["lora"] == {"r": 16, "alpha": 32, "dropout": 0.0} and c["seed"] == 7
        y = rd(f"xeval-yd3-dev-{init}-s7.json")["xeval"]
        assert y["data_name"] == "a6" and y["panels"] == ["yd3"] and y["seed"] == 9
        assert len(y["checkpoints"]) == 1 + 4 * 5 and all(k["attempts"] == 16 for k in y["checkpoints"])
        e = rd(f"xeval-rd3-dev-{init}-s7.json")["xeval"]
        assert e["checkpoints"][0]["attempts"] == 48 and e["checkpoints"][0]["split"] == 24
        assert all(k["attempts"] == 32 and k["adapter"].endswith("round0256.safetensors") for k in e["checkpoints"][1:])
        if PD and os.path.isdir(os.path.join(PD, "runs")):
            assert os.path.exists(os.path.join(PD, b["init"]["adapter"]))
