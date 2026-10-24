"""Tests for tools/bc2rl_guard.py (laptop-side; no Slurm access needed)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import bc2rl_guard as g  # noqa: E402

SQ = """\
1|wcw-P01|RUNNING|preempt|preempt_qos|gres/gpu:1|32|None
2|wcw-P05|PENDING|preempt|preempt_qos|gres/gpu:1|32|Priority
3|ae-geo-x|RUNNING|preempt|preempt_qos|gres/gpu:L40S:1|8|None
4|ae-geo-y|PENDING|preempt|preempt_qos|gres/gpu:L40S:2|8|Priority
5|ae-geo-g|RUNNING|general|normal|gres/gpu:L40S:1|8|None
6|wcw-ctl|RUNNING|preempt|preempt_cpu_qos|N/A|1|None
7|ae-bc2rl-a-s11|RUNNING|preempt|preempt_qos|gres/gpu:1|4|None
8|mystery|CONFIGURING|preempt|preempt_qos|gres/gpu:1|4|None
9|ae-bc2rl-cpu-s0|PENDING|preempt|preempt_cpu_qos|N/A|4|None
"""
LIM = "preempt_qos|gres/gpu=24\npreempt_cpu_qos|cpu=64\n"
OURS = {"7": "bc2rl-a-s11", "9": "bc2rl-cpu-s0"}
GPU = {"gpus": "1", "cpus": "4", "mem": "32G", "time": "08:00:00", "part": "preempt"}


def acc(res=()):
    L, cpu = g.parse_limits(LIM)
    return g.account(g.parse_squeue(SQ), L, cpu, OURS, list(res))


def test_gres_parsing():
    assert g.gpus_from_gres("gres/gpu:1") == 1
    assert g.gpus_from_gres("gres/gpu:L40S:2") == 2
    assert g.gpus_from_gres("N/A") == 0
    try:
        g.gpus_from_gres("gres/gpu:weird")
    except g.Unverifiable:
        pass
    else:
        raise AssertionError("unparseable gres must fail closed")


def test_classification_and_cap():
    a = acc()
    assert (a.W, a.A, a.U, a.ours_gpu) == (2, 3, 1, 1)       # pending GPUs count; general ignored
    assert a.cap == min(5, 24 - 9 - 8 - 1 - 2) == 4
    assert [j.jobid for j in a.ours_cpu_jobs] == ["9"]


def test_cap_formula_uses_actual_when_above_reservation():
    a = acc()
    a.A, a.W, a.U = 12, 8, 0
    assert a.cap == 24 - 12 - 8 - 0 - 2 == 2
    a.A = 30
    assert a.cap == 0


def test_gpu_submission_limits():
    a = acc()
    assert g.decide("bc2rl-base-dev-s0", GPU, a, [], OURS) == []
    a.ours_gpu = 4
    assert any("cap" in e for e in g.decide("bc2rl-base-dev-s0", GPU, a, [], OURS))
    a.ours_gpu = 5
    assert any("OVER CAP" in e for e in g.decide("bc2rl-base-dev-s0", GPU, a, [], OURS))


def test_reservations_count_until_visible():
    a = acc([{"jobid": None, "gpus": 1}, {"jobid": "7", "gpus": 1}, {"jobid": "99", "gpus": 1}])
    assert a.ours_gpu == 2          # 7 counted by squeue; 99 confirmed and finished; the unconfirmed adds 1


def test_recipe_rules():
    bad = dict(GPU, part="general")
    assert any("part=preempt" in e for e in g.check_recipe("bc2rl-a-b-s1", bad))
    assert any("8 h" in e for e in g.check_recipe("bc2rl-a-b-s1", dict(GPU, time="09:00:00")))
    assert any("one GPU" in e for e in g.check_recipe("bc2rl-a-b-s1", dict(GPU, gpus="2")))
    assert any("32G" in e for e in g.check_recipe("bc2rl-a-b-s1", dict(GPU, mem="64G")))
    assert any("CPUs" in e for e in g.check_recipe("bc2rl-a-b-s1", {"gpus": "0", "cpus": "8", "time": "01:00:00"}))
    assert any("expid" in e for e in g.check_recipe("myrun", GPU))
    no_part = {k: v for k, v in GPU.items() if k != "part"}
    assert any("part=preempt" in e for e in g.check_recipe("bc2rl-a-b-s1", no_part))


def test_cpu_one_concurrent_job_but_chain_allowed():
    a = acc()
    cpu = {"gpus": "0", "cpus": "4", "time": "01:00:00"}
    assert any("nonterminal CPU" in e for e in g.decide("bc2rl-x-y-s0", cpu, a, [], OURS))
    assert g.decide("bc2rl-x-y-s0", cpu, a, ["bc2rl-cpu-s0"], OURS) == []
