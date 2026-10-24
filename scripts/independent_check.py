#!/usr/bin/env python3
"""Independent re-scoring of evaluation records (shares no code with bc2rl.task.verify).

Algorithm: keep only lines containing '='; each must read "<int> <op> <int> = <int>" after removing
spaces and an optional list/step label; simulate with a plain list (list.remove). Success iff
exactly n-1 such lines, every line legal and correct, final list == [target], and not truncated.
Usage: independent_check.py <split.jsonl> <records.jsonl.gz> [...]   -> agreement report (JSON)
"""
import gzip
import json
import re
import sys


def score(inputs, target, text, finish):
    if finish == "length":
        return 0
    avail = list(inputs)
    steps = 0
    for raw in text.replace(";", "\n").splitlines():
        if "=" not in raw:
            continue
        s = raw.replace(" ", "").replace("×", "*").replace("−", "-").replace("`", "").replace("**", "").rstrip(".")
        s = re.sub(r"^(?:[-*•](?=\d)|\d+[.)]|step\d+[:.)])", "", s, flags=re.I) if not re.match(r"^-?\d+[-+*]", s) else s
        m = re.fullmatch(r"(-?\d+)([-+*])(-?\d+)=(-?\d+)", s)
        if not m:
            return 0
        a, op, b, c = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        for v in (a, b):
            if v not in avail:
                return 0
            avail.remove(v)
        if c != {"+": a + b, "-": a - b, "*": a * b}[op]:
            return 0
        avail.append(c)
        steps += 1
    return int(steps == len(inputs) - 1 and avail == [target])


def main():
    probs = {json.loads(l)["pid"]: json.loads(l) for l in open(sys.argv[1])}
    out = {}
    for path in sys.argv[2:]:
        agree = n = succ_a = succ_b = 0
        mism = []
        for line in gzip.open(path, "rt"):
            r = json.loads(line)
            p = probs[r["pid"]]
            b = score(p["inputs"], p["target"], r["text"], r["finish"])
            n += 1
            agree += int(b == r["reward"])
            succ_a += r["reward"]
            succ_b += b
            if b != r["reward"] and len(mism) < 10:
                mism.append({"pid": r["pid"], "verifier": r["reward"], "independent": b, "text": r["text"][:300]})
        out[path] = {"records": n, "agree": agree, "verifier_successes": succ_a, "independent_successes": succ_b,
                     "mismatches": mism}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
