"""python -m bc2rl {generate,bc,online,evaluate,report} --config <yaml> [--run-id ID] [--results-dir DIR]"""
from __future__ import annotations

import argparse
import os

from .runtime import load_config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bc2rl")
    ap.add_argument("command", choices=["generate", "bc", "online", "evaluate", "report"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", help="explicit run identifier; resume uses the same id (overrides config run_id)")
    ap.add_argument("--results-dir", help="directory for the small harvested distillate (e.g. $AE_RESULTS)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.run_id:
        cfg["run_id"] = a.run_id
    if a.results_dir:
        os.makedirs(a.results_dir, exist_ok=True)
        cfg["_results_dir"] = a.results_dir
    if a.command == "generate":
        from .data import cmd_generate
        return cmd_generate(cfg)
    if a.command == "bc":
        from .bc import cmd_bc
        return cmd_bc(cfg)
    if a.command == "online":
        from .online import cmd_online
        return cmd_online(cfg)
    if a.command == "evaluate":
        from .evaluate import cmd_evaluate
        return cmd_evaluate(cfg)
    if a.command == "report":
        from .report import cmd_report
        return cmd_report(cfg)
    return 2
