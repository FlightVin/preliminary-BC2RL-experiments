"""Arithmetic construction task: generator, frozen splits, prompts and the trace verifier.

A problem gives n positive integers and a target. A solution is exactly n-1 lines `a op b = c`
(op in + - *). Each line consumes a and b from the multiset of available values and adds c.
Success: every line is valid and, after n-1 lines, the only remaining value is the target.
Generated text is parsed, never executed.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, asdict, field

VERIFIER_VERSION = "arith-verifier-v1"
GENERATOR_VERSION = "arith-generator-v1"
OPS = ("+", "-", "*")


# ----------------------------------------------------------------------------- generator

@dataclass(frozen=True)
class GenConfig:
    min_value: int = 2
    max_value: int = 20
    max_target: int = 1000
    max_abs_intermediate: int = 10000
    ops: tuple = OPS


@dataclass
class Problem:
    pid: str
    split: str
    n_inputs: int
    inputs: list          # presentation order
    target: int
    ref_trace: list       # list of "a op b = c" strings (hidden from online learners)
    structure: str        # "chain" | "branch"
    ops_used: list = field(default_factory=list)

    @property
    def key(self):
        return canonical_key(self.inputs, self.target)

    def to_json(self):
        return asdict(self)


def canonical_key(inputs, target):
    return (tuple(sorted(int(x) for x in inputs)), int(target))


def apply_op(op, a, b):
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    raise ValueError(op)


class Reject(Exception):
    pass


def _build(rng, values, cfg, trace, ops_used):
    """Random full binary tree over `values` (in order). Returns (value, is_leaf)."""
    if len(values) == 1:
        return values[0], True
    k = rng.randint(1, len(values) - 1)
    lv, lleaf = _build(rng, values[:k], cfg, trace, ops_used)
    rv, rleaf = _build(rng, values[k:], cfg, trace, ops_used)
    op = rng.choice(cfg.ops)
    if op == "*" and (lv == 1 or rv == 1):
        raise Reject("multiplicative identity")
    v = apply_op(op, lv, rv)
    if v == 0:
        raise Reject("zero subexpression")
    if abs(v) > cfg.max_abs_intermediate:
        raise Reject("intermediate bound")
    trace.append((f"{lv} {op} {rv} = {v}", lleaf, rleaf))
    ops_used.append(op)
    return v, False


def sample_problem(rng: random.Random, n: int, cfg: GenConfig, stats: Counter):
    """Rejection-sample one problem with n inputs. Returns (inputs_unshuffled, target, trace, structure, ops)."""
    while True:
        values = [rng.randint(cfg.min_value, cfg.max_value) for _ in range(n)]
        trace, ops_used = [], []
        try:
            target, _ = _build(rng, values, cfg, trace, ops_used)
        except Reject as e:
            stats[f"reject:{e}"] += 1
            continue
        if not (1 <= target <= cfg.max_target):
            stats["reject:target bound"] += 1
            continue
        stats["accepted"] += 1
        # chain: every operation combines a running result with at most one new subtree leaf pair
        structure = "chain" if all(lleaf or rleaf for _, lleaf, rleaf in trace) else "branch"
        return values, target, [t for t, _, _ in trace], structure, ops_used


SPLIT_SIZES = {
    # split: (count per horizon) -- balanced across the three input counts as closely as possible
    "dev": 144, "devdemo": 512, "bc": 2048, "online": 4096, "val": 144, "test": 288,
}
HORIZONS = (3, 5, 10)


def per_horizon(total, horizons=HORIZONS):
    base, extra = divmod(total, len(horizons))
    return {h: base + (1 if i < extra else 0) for i, h in enumerate(horizons)}


def generate_splits(main_seed: int, dev_seed: int, cfg: GenConfig = GenConfig(), horizons=HORIZONS,
                    sizes=None, format_example_inputs: int = 4):
    """Generate disjoint splits. dev/devdemo/format example use dev_seed; the rest use main_seed.

    Deduplication key: (sorted input multiset, target), global across all splits.
    """
    sizes = dict(SPLIT_SIZES if sizes is None else sizes)
    seen = set()
    out = {}
    stats = {}

    def fill(split, rng, n, count):
        st = stats.setdefault(f"{split}:{n}", Counter())
        got = []
        while len(got) < count:
            values, target, trace, structure, ops_used = sample_problem(rng, n, cfg, st)
            key = canonical_key(values, target)
            if key in seen:
                st["reject:duplicate"] += 1
                continue
            seen.add(key)
            shown = values[:]
            rng.shuffle(shown)
            got.append(Problem(pid=f"{split}-{n}-{len(got):05d}", split=split, n_inputs=n, inputs=shown,
                               target=target, ref_trace=trace, structure=structure, ops_used=ops_used))
        return got

    drng = random.Random(dev_seed)
    fmt = fill("format", drng, format_example_inputs, 1)
    out["format"] = fmt
    for split in ("dev", "devdemo"):
        out[split] = []
        for h, c in per_horizon(sizes[split], horizons).items():
            out[split] += fill(split, drng, h, c)
    mrng = random.Random(main_seed)
    for split in ("bc", "online", "val", "test"):
        out[split] = []
        for h, c in per_horizon(sizes[split], horizons).items():
            out[split] += fill(split, mrng, h, c)
    return out, {k: dict(v) for k, v in stats.items()}


def manifest_hash(problems):
    h = hashlib.sha256()
    for p in problems:
        h.update(json.dumps(p.to_json() if isinstance(p, Problem) else p, sort_keys=True).encode())
        h.update(b"\n")
    return h.hexdigest()


def load_split(path):
    return [Problem(**json.loads(l)) for l in open(path) if l.strip()]


def save_split(path, problems):
    with open(path, "w") as f:
        for p in problems:
            f.write(json.dumps(p.to_json(), sort_keys=True) + "\n")


# ----------------------------------------------------------------------------- prompts

RULES = (
    "Combine the given numbers to reach the target.\n"
    "Rules:\n"
    "- Use every number exactly once.\n"
    "- Allowed operations: +, -, *.\n"
    "- Write one operation per line in the form \"a op b = c\". Both a and b must be numbers that "
    "are still available. The operation uses up a and b, and the result c becomes available.\n"
    "- Use exactly one operation fewer than the number of given numbers ({k} operations here). "
    "After the last operation, the only remaining number must be the target.\n"
    "- Write only the operation lines."
)


def problem_block(inputs, target):
    return f"Numbers: {', '.join(str(x) for x in inputs)}\nTarget: {target}"


def prompt_text(problem: Problem, variant: str, format_example: Problem | None):
    rules = RULES.format(k=problem.n_inputs - 1)
    if variant == "zeroshot":
        return f"{rules}\n\n{problem_block(problem.inputs, problem.target)}"
    if variant == "format":
        ex = format_example
        ex_text = f"{problem_block(ex.inputs, ex.target)}\n" + "\n".join(ex.ref_trace)
        return (f"{rules}\n\nExample:\n{ex_text}\n\nNow solve this one.\n"
                f"{problem_block(problem.inputs, problem.target)}")
    raise ValueError(variant)


def completion_text(problem: Problem):
    """BC target text (the chat template adds the end-of-turn token)."""
    return "\n".join(problem.ref_trace)


# ----------------------------------------------------------------------------- verifier

_OP_LINE = re.compile(r"^(-?\d+)\s*([+\-*×])\s*(-?\d+)\s*=\s*(-?\d+)\.?$")
_PREFIX = re.compile(r"^(?:[-*•]\s+|\d+[.)]\s+|step\s*\d+\s*[:.)]\s*)", re.IGNORECASE)


def _normalize(seg: str) -> str:
    s = seg.replace("−", "-").replace("–", "-").replace("`", "").replace("**", "").replace("$", "")
    s = s.strip()
    s = _PREFIX.sub("", s, count=1).strip()
    return s


def parse_trace(text: str):
    """Return (ops, malformed_count). ops: list of (a, op, b, c) ints/str.

    Segments are lines or ';'-separated parts. A segment that contains '=' must be one operation;
    otherwise it is a malformed claim. Segments without '=' are prose and are ignored.
    """
    ops, malformed = [], 0
    for line in text.replace("\r", "").split("\n"):
        for seg in line.split(";"):
            s = _normalize(seg)
            if not s:
                continue
            m = _OP_LINE.match(s)
            if m:
                op = "*" if m.group(2) == "×" else m.group(2)
                ops.append((int(m.group(1)), op, int(m.group(3)), int(m.group(4))))
            elif "=" in s:
                malformed += 1
    return ops, malformed


def verify(inputs, target, text: str, truncated: bool = False):
    """Score one completion. Returns a dict with reward (0/1), error category and diagnostics."""
    ops, malformed = parse_trace(text)
    avail = Counter(int(x) for x in inputs)
    n_needed = len(inputs) - 1
    valid_prefix = 0
    op_error = None
    for (a, op, b, c) in ops:
        if avail[a] <= 0:
            op_error = "illegal_operand"
            break
        avail[a] -= 1
        if avail[b] <= 0:
            op_error = "illegal_operand"
            break
        avail[b] -= 1
        if apply_op(op, a, b) != c:
            op_error = "arith_error"
            break
        avail[c] += 1
        valid_prefix += 1
    remaining = sorted(v for v, k in avail.items() for _ in range(k))
    format_valid = (malformed == 0 and len(ops) > 0)
    if truncated:
        error = "truncated"
    elif malformed:
        error = "malformed"
    elif not ops:
        error = "no_ops"
    elif op_error:
        error = op_error
    elif len(ops) < n_needed:
        error = "too_few_ops"
    elif remaining != [int(target)]:
        error = "wrong_final"
    else:
        error = None
    return {
        "reward": int(error is None),
        "error": error or "success",
        "format_valid": bool(format_valid),
        "n_ops": len(ops),
        "valid_prefix": valid_prefix,
        "verifier": VERIFIER_VERSION,
    }
