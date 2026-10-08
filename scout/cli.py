"""Shared command-line plumbing for the scripts in ``scripts/``."""

from __future__ import annotations

import argparse
from typing import Dict, List

from scout.configs import TaskConfig, TASKS, get_task_config


def _coerce(value: str):
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def parse_overrides(items: List[str]) -> Dict[str, object]:
    """``["train.epochs=10", "ttt_steps=5"]`` -> ``{"train.epochs": 10, "ttt_steps": 5}``."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"override must look like key=value, got '{item}'")
        k, v = item.split("=", 1)
        out[k.strip()] = _coerce(v.strip())
    return out


def add_task_args(parser: argparse.ArgumentParser):
    parser.add_argument("--task", required=True, choices=sorted(TASKS),
                        help="Task preset from scout.configs.TASKS")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Override config fields, e.g. --set train.epochs=10 model.k_components=2 "
                             "rollout.ttt_steps=5 (section prefix optional when unambiguous)")


def load_config(args) -> TaskConfig:
    cfg = get_task_config(args.task)
    return cfg.override(**parse_overrides(args.set))
