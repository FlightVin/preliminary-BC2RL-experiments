"""Exact solution structure of arithmetic problems (A6 program S, protocol Section 13.3).

Enumerates every valid solution under the verifier v1 rules (`task.verify`): each operation combines two available
numbers with +, - or *; the result becomes available; after n - 1 operations the only remaining number is the target.
Negative, zero and large intermediates are legal. Three levels are counted separately:
  trace  ordered operation chain ((a, op, b, c), ...); operand order of + and * is collapsed
  form   canonical expression over the input values: + and - flattened to a signed multiset of terms, * flattened to a
         multiset of factors, sorted recursively; (a+b)-c and (a-c)+b are one form
  shape  a form with every value replaced by the placeholder x, e.g. x*x+x
"""
from __future__ import annotations

from .task import apply_op

LEAF = "v"


def _terms(e):
    return e[1] if e[0] == "+" else ((1, e),)


def _factors(e):
    return e[1] if e[0] == "*" else (e,)


def _key(e):
    return repr(e)


def combine(x, op, y):
    """Canonical expression of (x op y) for canonical expressions x, y."""
    if op == "*":
        return ("*", tuple(sorted(_factors(x) + _factors(y), key=_key)))
    sy = 1 if op == "+" else -1
    terms = _terms(x) + tuple((sy * s, t) for s, t in _terms(y))
    return ("+", tuple(sorted(terms, key=_key)))


def leaf(v):
    return (LEAF, int(v))


def shape_of(e):
    if e[0] == LEAF:
        return (LEAF, "x")
    if e[0] == "*":
        return ("*", tuple(sorted((shape_of(f) for f in e[1]), key=_key)))
    return ("+", tuple(sorted(((s, shape_of(t)) for s, t in e[1]), key=_key)))


def render(e):
    """Readable string of a canonical expression or shape."""
    if e[0] == LEAF:
        return str(e[1])
    if e[0] == "*":
        return "*".join(render(f) if f[0] != "+" else f"({render(f)})" for f in e[1])
    out = ""
    for i, (s, t) in enumerate(sorted(e[1], key=lambda st: (-st[0], _key(st[1])))):
        r = render(t)
        out += (("" if i == 0 else "+") if s > 0 else "-") + r
    return out


def ops_of(e):
    if e[0] == LEAF:
        return set()
    sub = set().union(*(ops_of(f) for f in (e[1] if e[0] == "*" else (t for _, t in e[1]))))
    if e[0] == "*":
        return sub | {"*"}
    neg = sum(s < 0 for s, _ in e[1])
    return sub | ({"-"} if neg else set()) | ({"+"} if len(e[1]) - neg >= 2 else set())


def solutions(inputs, target):
    """Every valid solution. Returns a sorted list of unique (trace, form) pairs; trace = tuple of (a, op, b, c).

    One trace can carry two forms when an intermediate equals an input value (e.g. 5-2=3 next to an input 3)."""
    target = int(target)
    found = {}

    def rec(items, trace):
        if len(items) == 1:
            if items[0][0] == target:
                found[(tuple(trace), _key(items[0][1]))] = (tuple(trace), items[0][1])
            return
        n = len(items)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                (a, ea), (b, eb) = items[i], items[j]
                rest = [items[k] for k in range(n) if k not in (i, j)]
                for op in ("+", "-", "*"):
                    if op in "+*" and (a, _key(ea)) > (b, _key(eb)):
                        continue  # commutative: one operand order
                    c = apply_op(op, a, b)
                    step = (min(a, b), op, max(a, b), c) if op in "+*" else (a, op, b, c)
                    rec(rest + [(c, combine(ea, op, eb))], trace + [step])

    rec([(int(v), leaf(v)) for v in inputs], [])
    return [found[k] for k in sorted(found)]


def labels(inputs, target):
    """Problem labels of protocol 13.3 from the complete solution set."""
    sols = solutions(inputs, target)
    forms = {_key(f): f for _, f in sols}
    shapes = {_key(shape_of(f)): shape_of(f) for f in forms.values()}
    form_ops = [ops_of(f) for f in forms.values()]
    n_mult = [sum(op == "*" for _, op, _, _ in t) for t, _ in sols]
    neg = [any(c < 0 for _, _, _, c in t[:-1]) for t, _ in sols]
    zero = [any(c == 0 for _, _, _, c in t[:-1]) for t, _ in sols]
    return {
        "n_traces": len({t for t, _ in sols}),
        "n_forms": len(forms),
        "n_shapes": len(shapes),
        "forms": sorted(render(f) for f in forms.values()),
        "shapes": sorted(render(s) for s in shapes.values()),
        "cls": None if not sols else ("A" if any(o <= {"+", "-"} for o in form_ops) else "B"),
        "min_mult": min(n_mult) if sols else None,
        "neg_needed": bool(sols) and all(neg),
        "zero_needed": bool(sols) and all(zero),
        "input_max": max(int(v) for v in inputs),
        "input_sum": sum(int(v) for v in inputs),
        "target": int(target),
    }


def form_of_trace(inputs, steps):
    """Canonical form of a parsed, verified operation chain [(a, op, b, c), ...]; None if the chain is not valid.

    When a value is available more than once, the most recently produced one is used (an intermediate before an input)."""
    pool = [(int(v), leaf(v)) for v in inputs]
    pick = lambda x: next((k for k in range(len(pool) - 1, -1, -1) if pool[k][0] == x), None)
    for a, op, b, c in steps:
        ia = pick(a)
        if ia is None:
            return None
        ea = pool.pop(ia)[1]
        ib = pick(b)
        if ib is None or apply_op(op, a, b) != c:
            return None
        eb = pool.pop(ib)[1]
        pool.append((c, combine(ea, op, eb)))
    return render(pool[0][1]) if len(pool) == 1 else None
