"""E0/E2 tests: exact Adam displacement, matching, search, toy model, and a tiny-model end-to-end run (CPU)."""
import json
import os
import subprocess
import sys
from collections import Counter

import numpy as np
import pytest
import torch

from bc2rl.mech import Searcher, adam_first_step, apportion, cos, make_pools

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PD = os.environ.get("PROJ_DATA", "")


@pytest.mark.parametrize("scale", [1e-6, 10.0])
def test_adam_first_step_matches_torch_adamw(scale):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(1000))           # zero start: p - p0 is the update without cancellation
    g = torch.randn(1000) * scale
    g[:10] = 0.0
    p0 = p.detach().clone()
    opt = torch.optim.AdamW([p], lr=1e-5, weight_decay=0.0)
    p.grad = g.clone()
    torch.nn.utils.clip_grad_norm_([p], 1.0)
    opt.step()
    d = adam_first_step(g, 1e-5)
    assert torch.allclose(p.detach() - p0, d, atol=1e-11, rtol=1e-4)


def test_apportion_exact_and_capped():
    counts = {(3, 1): 9, (3, 2): 6, (5, 1): 3, (3, 3): 1}
    c = apportion(counts, 8)
    assert sum(c.values()) == 8
    assert all(v <= counts[k] // 2 for k, v in c.items())
    assert apportion({(3, 1): 9, (3, 2): 5, (5, 1): 2, (3, 3): 1}, 8) is None     # caps 4+2+1+0 = 7 < 8
    assert apportion({(3, 1): 5, (3, 2): 3}, 8) is None


def test_pools_disjoint_deterministic():
    keys = [(3, 1)] * 40 + [(3, 2)] * 30 + [(5, 1)] * 10
    a, b = make_pools(keys, 8, 16, 777), make_pools(keys, 8, 16, 777)
    assert a == b and len(a) == 5
    flat = [i for p in a for i in p]
    assert len(flat) == len(set(flat)) == 80


def test_search_matches_histograms_and_improves():
    torch.manual_seed(1)
    n, D = 40, 500
    G = (torch.randn(n, D) * 1e-3).to(torch.bfloat16)
    h = torch.randn(D)
    keys = [(3, 1 + (i % 3)) for i in range(n)]
    lens = [50.0 + (i % 5) for i in range(n)]
    pool = list(range(n))
    comp = apportion(Counter(keys), 8)
    single = {i: cos(h, adam_first_step(G[i].float(), 1e-5)) for i in range(n)}
    s = Searcher(G, h, lens, 1e-5, 0.05, 2000)
    H0 = sorted(pool, key=lambda i: -single[i])
    H, L, info = s.run(pool, keys, comp, single)
    assert Counter(keys[i] for i in H) == Counter(keys[i] for i in L)
    assert not set(H) & set(L) and len(H) == len(L) == 8
    assert info["search_align_H"] > info["search_align_L"]
    assert info["len_violation"] <= 0.05


def test_toy_checks(tmp_path):
    cfg = {"run_id": "toy", "proj_data": str(tmp_path), "seed": 0}
    p = tmp_path / "toy.yaml"
    p.write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, "-m", "bc2rl", "toy", "--config", str(p), "--results-dir", str(tmp_path / "res")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert json.load(open(tmp_path / "res" / "toy_checks.json"))["all_ok"]


@pytest.mark.skipif(not os.path.exists(os.path.join(PD, "model_revision.json")), reason="needs PROJ_DATA tokenizer")
def test_mech_end_to_end_tiny(tmp_path):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    pd = tmp_path / "pd"
    (pd / "data" / "t1").mkdir(parents=True)
    json.dump(json.load(open(os.path.join(PD, "model_revision.json"))), open(pd / "model_revision.json", "w"))
    torch.manual_seed(0)
    Qwen2ForCausalLM(Qwen2Config(vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                 num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                                 tie_word_embeddings=True)).save_pretrained(tmp_path / "tiny")
    base = {"proj_data": str(pd), "device": "cpu", "test_mode": True, "seed": 11, "prompt_variant": "format",
            "model_path_override": str(tmp_path / "tiny"), "model": {"lora": {"r": 4, "alpha": 8, "dropout": 0.0}},
            "data": {"version": "t1", "main_seed": 1, "dev_seed": 2, "min_value": 2, "max_value": 20,
                     "max_target": 1000, "max_abs_intermediate": 10000, "ops": ["+", "-", "*"], "horizons": [3, 5, 10],
                     "sizes": {"dev": 6, "devdemo": 6, "bc": 6, "online": 120, "val": 3, "test": 3}}}

    def cli(cmd, cfg, expect=0):
        p = tmp_path / f"{cfg['run_id']}.yaml"
        p.write_text(json.dumps(cfg))
        r = subprocess.run([sys.executable, "-m", "bc2rl", cmd, "--config", str(p), "--results-dir",
                            str(tmp_path / "res" / cfg["run_id"])], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == expect, r.stdout[-3000:] + r.stderr[-3000:]
        return r
    cli("generate", dict(base, run_id="gen"))
    cli("mechdata", dict(base, run_id="md", mechdata={"seed": 5, "per_inputs": 2}))
    # LoRA adapter to start from: a fresh adapter saved by the package itself
    from bc2rl.modeling import load_policy, save_adapter
    m = load_policy(str(tmp_path / "tiny"), base["model"]["lora"], seed=0, device="cpu")
    with torch.no_grad():
        for n_, p_ in m.named_parameters():
            if p_.requires_grad:
                p_.add_(0.01 * torch.randn_like(p_))
    save_adapter(m, str(pd / "runs" / "init" / "a.safetensors"))
    me = {"group": 4, "cap": 6, "micro": 4, "lr": 1e-3, "groups_per_arm": 2, "pool_prompts": 120, "pool_seed": 1,
          "max_mixed": 64, "max_pools": 8, "pool_split_seed": 777, "len_tol": 1.0, "search_budget": 200, "eval_seed": 1,
          "probe_attempts": 2, "kl_samples": 1, "batch_seqs": 64, "adapter": "runs/init/a.safetensors",
          "source_splits": ["online"], "sample_seed": 3, "n_pairs": 1, "debug_reward": "random"}
    cli("mech", dict(base, run_id="cal", mech=dict(me, mode="calibrate", kl_split="dev", kl_target=10.0, rho_grid=[1, 2])))
    cal = json.load(open(pd / "runs" / "cal" / "calibration.json"))
    assert cal["chosen_rho"] == 2 and len(cal["grid"]) == 2
    cli("mech", dict(base, run_id="iv", mech=dict(me, mode="intervene", rho=2)))
    s = json.load(open(pd / "runs" / "iv" / "summary.json"))
    assert set(s["forks"]) == {"base", "p0-H", "p0-L"}
    pr = s["pairs_used"][0]
    assert pr["hist_H"] == pr["hist_L"]
    assert abs(pr["exact"]["H"]["d_norm_equal"] - pr["exact"]["L"]["d_norm_equal"]) < 1e-6 * pr["exact"]["H"]["d_norm_equal"] + 1e-12
    for t in ("p0-H", "p0-L"):
        f = s["forks"][t]
        assert f["kl_median"] >= -1e-6 and np.isfinite(f["taylor_pred_fit"])
        assert sum(f["probe"]["n"]) == 6 * 2
    cli("mech", dict(base, run_id="iv", mech=dict(me, mode="intervene", rho=2)))     # resume: already complete
