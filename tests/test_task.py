import random
from collections import Counter

from bc2rl import task
from bc2rl.task import verify


def test_example_from_proposal():
    assert verify([2, 3, 4], 20, "2 + 3 = 5\n5 * 4 = 20")["reward"] == 1


def test_number_multiplicity():
    assert verify([3, 3, 2], 18, "3 * 3 = 9\n9 * 2 = 18")["reward"] == 1
    r = verify([3, 2, 4], 18, "3 * 3 = 9\n9 * 2 = 18")      # only one 3 available
    assert r["reward"] == 0 and r["error"] == "illegal_operand"


def test_subtraction_order_and_negative_intermediates():
    assert verify([5, 8, 4], 12, "5 - 8 = -3\n-3 + 4 = 1")["error"] == "wrong_final"
    assert verify([5, 8, 4], 1, "5 - 8 = -3\n-3 + 4 = 1")["reward"] == 1
    assert verify([5, 8, 4], 1, "5 - 8 = 3\n3 - 4 = -1")["error"] == "arith_error"
    assert verify([5, 8, 4], 12, "8 - 5 = 3\n3 * 4 = 12")["reward"] == 1


def test_illegal_reuse_of_result_and_input():
    r = verify([2, 3, 4], 20, "2 + 3 = 5\n5 * 4 = 20\n20 + 2 = 22")
    assert r["reward"] == 0 and r["error"] == "illegal_operand"
    assert verify([2, 3, 4], 25, "2 + 3 = 5\n5 * 5 = 25")["error"] == "illegal_operand"


def test_malformed_and_prose():
    assert verify([2, 3, 4], 20, "2 + 3 + 4 = 9\n9 * 2 = 18")["error"] == "malformed"
    assert verify([2, 3, 4], 20, "2 / 2 = 1")["error"] == "malformed"
    r = verify([2, 3, 4], 20, "Here is my solution:\n2 + 3 = 5\n5 * 4 = 20\nDone.")
    assert r["reward"] == 1 and r["format_valid"]
    assert verify([2, 3, 4], 20, "I cannot solve this.")["error"] == "no_ops"
    assert verify([2, 3, 4], 20, "2 + 3 = 5\n5 * 4 = 20\nSo the answer = 20")["error"] == "malformed"


def test_formatting_variants():
    assert verify([2, 3, 4], 20, "1. 2 + 3 = 5\n2. 5 × 4 = 20")["reward"] == 1
    assert verify([2, 3, 4], 20, "- 2 + 3 = 5\n- 5 * 4 = 20")["reward"] == 1
    assert verify([2, 3, 4], 20, "Step 1: 2 + 3 = 5; Step 2: 5 * 4 = 20")["reward"] == 1
    assert verify([2, 3, 4], 20, "  2+3=5 \n 5*4=20.")["reward"] == 1


def test_alternate_valid_solution_accepted():
    # 4 * 5 = 20 is not the reference construction but is valid
    assert verify([2, 3, 4], 20, "3 + 2 = 5\n4 * 5 = 20")["reward"] == 1


def test_counts_truncation_and_prefix():
    r = verify([2, 3, 4, 5], 100, "2 + 3 = 5\n5 * 4 = 20")
    assert r["error"] == "too_few_ops" and r["valid_prefix"] == 2
    t = verify([2, 3, 4], 20, "2 + 3 = 5\n5 * 4 = 20", truncated=True)
    assert t["reward"] == 0 and t["error"] == "truncated"
    assert verify([2, 3, 4], 20, "2 + 3 = 5\n5 * 4 = 21")["valid_prefix"] == 1


def test_generator_constraints_and_reference_traces():
    cfg = task.GenConfig()
    rng = random.Random(0)
    st = Counter()
    for n in (3, 5, 10):
        for _ in range(200):
            vals, target, trace, structure, ops = task.sample_problem(rng, n, cfg, st)
            assert len(vals) == n and all(2 <= v <= 20 for v in vals)
            assert 1 <= target <= 1000 and len(trace) == n - 1 and len(ops) == n - 1
            assert verify(vals, target, "\n".join(trace))["reward"] == 1
            for line in trace:
                a, op, b, _, c = line.split()
                assert int(c) != 0 and abs(int(c)) <= 10000
                if op == "*":
                    assert int(a) != 1 and int(b) != 1
            assert structure in ("chain", "branch")
    assert st["accepted"] == 600


def test_chain_structure_label():
    rng = random.Random(1)
    st = Counter()
    seen = Counter()
    for _ in range(300):
        seen[task.sample_problem(rng, 5, task.GenConfig(), st)[3]] += 1
    assert seen["chain"] > 0 and seen["branch"] > 0
    assert task.sample_problem(random.Random(2), 3, task.GenConfig(), st)[3] == "chain"


def test_splits_disjoint_balanced_and_deterministic():
    sizes = {"dev": 12, "devdemo": 9, "bc": 30, "online": 30, "val": 12, "test": 24}
    a, _ = task.generate_splits(5, 6, sizes=sizes)
    b, _ = task.generate_splits(5, 6, sizes=sizes)
    assert task.manifest_hash(a["test"]) == task.manifest_hash(b["test"])
    keys = [p.key for s in a.values() for p in s]
    assert len(keys) == len(set(keys))
    assert Counter(p.n_inputs for p in a["test"]) == {3: 8, 5: 8, 10: 8}
    assert len(a["format"]) == 1 and a["format"][0].n_inputs == 4
    for s in a.values():
        for p in s:
            assert verify(p.inputs, p.target, "\n".join(p.ref_trace))["reward"] == 1


def test_prompts_hide_reference_and_differ_by_variant():
    sizes = {"dev": 3, "devdemo": 3, "bc": 3, "online": 3, "val": 3, "test": 3}
    sp, _ = task.generate_splits(1, 2, sizes=sizes)
    p, fmt = sp["online"][2], sp["format"][0]
    zs, fm = task.prompt_text(p, "zeroshot", fmt), task.prompt_text(p, "format", fmt)
    assert "Example" not in zs and "Example" in fm
    for line in p.ref_trace:
        assert line not in fm
    assert f"({p.n_inputs - 1} operations here)" in fm
