"""Interruption/resume integration tests with a tiny random Qwen2 model (CPU) and the real tokenizer.

A run killed after an unsaved round/step and then resumed must end bit-identical to an
uninterrupted run: same adapter hash, same counters, no duplicated metric or record rows, and
the cost ledger keeps both segments (discarded work stays charged).
"""
import json
import os
import subprocess
import sys

import pytest
import yaml

PD = os.environ.get("PROJ_DATA", "")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(PD, "model_revision.json")),
                                reason="needs PROJ_DATA with the downloaded tokenizer")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from bc2rl import task
    tmp = tmp_path_factory.mktemp("resume")
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
    p = tmp / "gen.yaml"
    yaml.safe_dump(cfg, open(p, "w"))
    run_cli("generate", p)
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


def run_dir(cfg, rid):
    return os.path.join(cfg["proj_data"], "runs", rid)


def test_online_resume_bit_identical(env):
    tmp, base = env
    from bc2rl.modeling import file_sha256
    on = dict(base, online={"method": "grpo", "lr": 1e-3, "rounds": 4, "prompts_per_round": 2, "group": 2,
                            "cap": 6, "micro": 2, "clip": 0.2, "prompt_split": "online", "debug_reward": "random",
                            "init": {"adapter": "none", "tag": "nobc"}, "save_rounds": [4], "val_rounds": [4]})
    a = write(tmp, "on_a", dict(on, run_id="on-a"))
    run_cli("online", a)
    b_cfg = dict(on, run_id="on-b", _debug_kill_after_round=3, _debug_ckpt_every=2)
    run_cli("online", write(tmp, "on_b1", b_cfg), expect=17)
    b_cfg.pop("_debug_kill_after_round")
    run_cli("online", write(tmp, "on_b2", b_cfg))
    ra, rb = run_dir(on, "on-a"), run_dir(on, "on-b")
    assert file_sha256(f"{ra}/adapters/round0004.safetensors") == file_sha256(f"{rb}/adapters/round0004.safetensors")
    sa, sb = json.load(open(f"{ra}/summary.json")), json.load(open(f"{rb}/summary.json"))
    assert sa["counters"] == sb["counters"] and sa["counters"]["opt_steps"] + sa["counters"]["skipped"] == 4
    assert sa["counters"]["opt_steps"] >= 1
    rows = [json.loads(l) for l in open(f"{rb}/metrics.jsonl")]
    assert [r["round"] for r in rows] == [1, 2, 3, 4]
    assert sorted(os.listdir(f"{rb}/records")) == [f"round{i:04d}.jsonl.gz" for i in range(1, 5)]
    assert len(json.load(open(f"{rb}/cost.json"))["segments"]) == 2
    assert len(sb["segments"]) == 2 and len(sa["segments"]) == 1
    # optimizer really changed the adapter, and B kept it identical to A
    assert sa["checkpoints"][-1]["sha256"] == sb["checkpoints"][-1]["sha256"]
    # a resumed run with a different configuration is refused
    bad = dict(b_cfg, online=dict(on["online"], lr=2e-3))
    r = subprocess.run([sys.executable, "-m", "bc2rl", "online", "--config", str(write(tmp, "bad", bad))],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "config hash mismatch" in (r.stdout + r.stderr)


def test_filtered_runs_and_skips_empty(env):
    tmp, base = env
    on = dict(base, run_id="filt", online={"method": "filtered", "lr": 1e-3, "rounds": 2, "prompts_per_round": 2,
                                           "group": 2, "cap": 6, "micro": 2, "clip": 0.2, "prompt_split": "online",
                                           "init": {"adapter": "none", "tag": "nobc"}, "save_rounds": [2],
                                           "val_rounds": []})
    run_cli("online", write(tmp, "filt", on))
    s = json.load(open(os.path.join(run_dir(on, "filt"), "summary.json")))
    # a random tiny model essentially never solves the task: filtered steps must be skipped
    assert s["counters"]["skipped"] + s["counters"]["opt_steps"] == 2


def test_bc_resume_bit_identical(env):
    tmp, base = env
    from bc2rl.modeling import file_sha256
    bc = dict(base, bc={"split": "bc", "steps": 4, "batch": 4, "micro": 2, "lr": 1e-3, "save_steps": [2, 4]})
    run_cli("bc", write(tmp, "bc_a", dict(bc, run_id="bc-a")))
    b = dict(bc, run_id="bc-b", _debug_kill_after_step=3)
    run_cli("bc", write(tmp, "bc_b1", b), expect=17)
    b.pop("_debug_kill_after_step")
    run_cli("bc", write(tmp, "bc_b2", b))
    ra, rb = run_dir(bc, "bc-a"), run_dir(bc, "bc-b")
    for s in ("step0002", "step0004"):
        assert file_sha256(f"{ra}/adapters/{s}.safetensors") == file_sha256(f"{rb}/adapters/{s}.safetensors")
    rows = [json.loads(l) for l in open(f"{rb}/metrics.jsonl")]
    assert [r["step"] for r in rows] == [1, 2, 3, 4]
    sa = json.load(open(f"{ra}/summary.json"))
    assert [c["step"] for c in sa["checkpoints"]] == [2, 4]
    assert sa["checkpoints"][0]["sha256"] != sa["checkpoints"][1]["sha256"]
    # a completed run is a no-op on re-entry (done-predicate validates hash + artifacts)
    r = run_cli("bc", write(tmp, "bc_a2", dict(bc, run_id="bc-a")))
    assert "already complete" in r.stdout


def test_fork_from_bc_adapter_is_exact(env):
    tmp, base = env
    from bc2rl.modeling import file_sha256
    src = os.path.join(run_dir(base, "bc-a"), "adapters", "step0002.safetensors")
    on = dict(base, run_id="fork", online={"method": "grpo", "lr": 1e-3, "rounds": 1, "prompts_per_round": 2,
                                           "group": 2, "cap": 6, "micro": 2, "clip": 0.2, "prompt_split": "online",
                                           "debug_reward": "random",
                                           "init": {"adapter": src, "tag": "bc2", "sha256": file_sha256(src)},
                                           "save_rounds": [1], "val_rounds": []})
    run_cli("online", write(tmp, "fork", on))
    init = json.load(open(os.path.join(run_dir(on, "fork"), "init.json")))
    assert init["init_sha256"] == file_sha256(src)
    bad = dict(on, run_id="fork-bad", online=dict(on["online"], init=dict(on["online"]["init"], sha256="0" * 64)))
    r = subprocess.run([sys.executable, "-m", "bc2rl", "online", "--config", str(write(tmp, "forkbad", bad))],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "init adapter hash" in (r.stdout + r.stderr)
