"""Tests for A6 program S stage S2 (protocol Section 13.3): exact solution enumeration (`bc2rl.structure`) and the
candidate / summary helpers of `scripts/s2_data.py`. Expected values are worked out by hand in the comments."""
import os
import random
import sys
from collections import Counter

from bc2rl import structure as S
from bc2rl import task
from bc2rl import yanalysis as Y

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import s2_data  # noqa: E402


def test_two_forms_mult_required():
    # (2, 3, 4) -> 10: 2*3+4 and 3*4-2 only; no + / - only chain reaches 10 (2+3+4 = 9)
    lab = S.labels([2, 3, 4], 10)
    assert lab["forms"] == ["2*3+4", "3*4-2"]
    assert lab["n_traces"] == 2 and lab["n_forms"] == 2
    assert lab["cls"] == "B" and lab["min_mult"] == 1
    assert lab["shapes"] == ["x*x+x", "x*x-x"]
    assert not lab["neg_needed"]


def test_associative_traces_share_one_form():
    # (7, 4, 2) -> 9: (4+7)-2, (7-2)+4, (4-2)+7 are three traces of the one form 4+7-2
    lab = S.labels([7, 4, 2], 9)
    assert "4+7-2" in lab["forms"] and lab["cls"] == "A" and lab["min_mult"] == 0
    assert lab["n_traces"] >= 3
    tr = {t for t, f in S.solutions([7, 4, 2], 9) if S.render(f) == "4+7-2"}
    assert {((4, "+", 7, 11), (11, "-", 2, 9)), ((7, "-", 2, 5), (4, "+", 5, 9)), ((4, "-", 2, 2), (2, "+", 7, 9))} <= tr
    assert S.form_of_trace([7, 4, 2], [(7, "+", 4, 11), (11, "-", 2, 9)]) == "4+7-2"
    assert S.form_of_trace([7, 4, 2], [(7, "-", 2, 5), (5, "+", 4, 9)]) == "4+7-2"


def test_operand_swap_and_input_order_invariance():
    a = S.labels([2, 3, 4], 10)
    for perm in ([4, 3, 2], [3, 2, 4], [4, 2, 3]):
        b = S.labels(perm, 10)
        assert (a["forms"], a["n_traces"], a["cls"]) == (b["forms"], b["n_traces"], b["cls"])
    assert S.form_of_trace([2, 3, 4], [(3, "*", 2, 6), (4, "+", 6, 10)]) == "2*3+4"
    assert S.form_of_trace([2, 3, 4], [(2, "*", 3, 6), (6, "+", 4, 10)]) == "2*3+4"


def test_invalid_chain_and_unsolvable():
    assert S.form_of_trace([2, 3, 4], [(2, "*", 3, 7), (7, "+", 4, 11)]) is None   # arithmetic error
    assert S.form_of_trace([2, 3, 4], [(5, "+", 3, 8)]) is None                    # illegal operand
    lab = S.labels([2, 2, 2], 1000)
    assert lab["n_traces"] == 0 and lab["cls"] is None and lab["min_mult"] is None


def test_most_recent_value_is_used_for_repeated_values():
    # 4 inputs (5, 2, 3, 4): 5-2=3 makes a second 3; the next 3*4 uses the intermediate
    assert S.form_of_trace([5, 2, 3, 4], [(5, "-", 2, 3), (3, "*", 4, 12), (12, "+", 3, 15)]) == "(5-2)*4+3"


def test_agrees_with_existing_3_input_enumeration():
    # trace count = yanalysis.solution_forms chain count; class B <=> every (op1, op2, side) form uses *
    rng, st = random.Random(7), Counter()
    for _ in range(300):
        values, target, *_ = task.sample_problem(rng, 3, task.GenConfig(), st)
        n_chains, old = Y.solution_forms(values, target)
        lab = S.labels(values, target)
        assert lab["n_traces"] == n_chains
        assert (lab["cls"] == "B") == all("*" in f[:2] for f in old)
        assert lab["n_forms"] >= 1 and lab["n_forms"] <= lab["n_traces"]


def test_ops_of_and_shapes():
    f = S.combine(S.combine(S.leaf(5), "-", S.leaf(2)), "*", S.leaf(4))
    assert S.ops_of(f) == {"-", "*"} and S.render(f) == "(5-2)*4" and S.render(S.shape_of(f)) == "(x-x)*x"
    g = S.combine(S.combine(S.leaf(2), "*", S.leaf(3)), "*", S.leaf(4))
    assert S.render(g) == "2*3*4" and S.ops_of(g) == {"*"}
    h = S.combine(S.combine(S.leaf(7), "+", S.leaf(4)), "-", S.leaf(2))
    assert S.ops_of(h) == {"+", "-"}


def test_generator_universe_small():
    # values 2..3 only: triples (2,2,2), (2,2,3), (2,3,3), (3,3,3); every target is generator-valid and in [1, 1000]
    cfg = task.GenConfig(min_value=2, max_value=3)
    uni = s2_data.universe(cfg)
    assert ((2, 2, 2), 6) in uni and ((2, 2, 2), 8) in uni and ((3, 3, 3), 27) in uni
    assert ((2, 2, 2), 2) in uni           # 2*2-2
    assert ((2, 2, 2), 0) not in uni and all(1 <= t <= 1000 for _, t in uni)
    assert ((2, 2, 3), 1) in uni           # 2*2-3 (no zero subexpression, no *1)


def test_draw_candidates_and_summary():
    seen = set()
    cands, stats = s2_data.draw_candidates(20261028, 2000, task.GenConfig(), seen, role_seed=1)
    assert stats["draws"] == 2000 and len(cands) == stats["unique_new"] and stats["duplicate_within"] + len(cands) == 2000
    assert len({c["key"] for c in cands}) == len(cands)
    assert {c["role"] for c in cands} <= {"pool", "dev_panel", "conf_panel"}
    again, _ = s2_data.draw_candidates(20261028, 2000, task.GenConfig(), set(), role_seed=1)
    assert [c["pid"] for c in again] == [c["pid"] for c in cands] and [c["role"] for c in again] == [c["role"] for c in cands]
    excl, st2 = s2_data.draw_candidates(20261028, 2000, task.GenConfig(), {cands[0]["key"]}, role_seed=1)
    assert st2["excluded_existing"] >= 1 and cands[0]["key"] not in {c["key"] for c in excl}
    summ = s2_data.summarize(cands)
    assert sum(summ["by_role_class"]["all"].values()) == len(cands)


import s2_build  # noqa: E402


def _synth_cands(n=400, seed=3):
    rng, out = random.Random(seed), []
    for i in range(n):
        cl = "A" if i % 3 else "B"
        out.append({"pid": f"st1-3-{i:06d}", "role": ("pool", "dev_panel", "conf_panel")[i % 4 % 3], "first_draw": i,
                    "inputs": [2, 3, 4], "target": rng.choice([5, 50, 500]), "ref_trace": [], "structure": "chain",
                    "ops_used": [], "lab": {"cls": cl, "n_forms": rng.choice([1, 2, 3]), "n_traces": 3, "input_max": 4,
                                            "input_sum": 9, "min_mult": int(cl == "B")}})
    return out


def test_s2_build_prescreen_cem_and_diag():
    cands = _synth_cands()
    sel = s2_build.select_prescreen(cands, 20, 10)
    assert Counter(c["lab"]["cls"] for c in sel["pre_pool"]) == {"A": 20, "B": 20}
    assert all(c["role"] == "pool" for c in sel["pre_pool"]) and all(c["role"] == "conf_panel" for c in sel["pre_conf"])
    rng = random.Random(5)
    succ = {c["pid"]: (rng.choice([0, 3, 9, 16]), rng.choice([0, 3, 9, 16])) for c in cands}
    pool = [c for c in cands if c["role"] == "pool"]
    trA, trB, rep = s2_build.cem_pools(pool, succ, 10**6, seed=1)
    assert len(trA) == len(trB) == rep["matched_pairs_total"] > 0
    assert all(c["lab"]["cls"] == "A" for c in trA) and all(c["lab"]["cls"] == "B" for c in trB)
    cell = lambda c: (s2_build.binned(succ[c["pid"]][0], s2_build.SUCC_BINS), s2_build.binned(succ[c["pid"]][1], s2_build.SUCC_BINS),
                      s2_build.binned(c["lab"]["n_forms"], s2_build.FORM_BINS), s2_build.binned(c["target"], s2_build.TARGET_BINS))
    assert all(cell(a) == cell(b) for a, b in zip(trA, trB))        # every prefix stays matched cell by cell
    assert rep["unmatched_A"] + rep["matched_pairs_total"] == sum(c["lab"]["cls"] == "A" for c in pool)
    trA2, _, _ = s2_build.cem_pools(pool, succ, 5, seed=1)
    assert [c["pid"] for c in trA2] == [c["pid"] for c in trA[:5]]
    diag, tab = s2_build.diag_panel([c for c in cands if c["role"] == "dev_panel"], succ, 3, seed=2)
    cnt = Counter((c["lab"]["cls"], s2_build.binned((succ[c["pid"]][0] + succ[c["pid"]][1]) / 32, s2_build.RATE_BINS)) for c in diag)
    assert max(cnt.values()) <= 3 and all(t["taken"] == min(3, t["available"]) for t in tab)
