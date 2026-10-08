"""Registry of Meta-World task definitions.

Task modules import the custom Meta-World policies, so they are loaded
lazily: ``get_task("pick_bar")`` imports only what that task needs.
"""

import importlib

from scout.envs.metaworld.tasks.base import MetaWorldTask

_MODULES = {
    "pick_bar": ("scout.envs.metaworld.tasks.pick_bar", "PickBarTask"),
    "push_bar": ("scout.envs.metaworld.tasks.push_bar", "PushBarTask"),
    "faucet": ("scout.envs.metaworld.tasks.faucet", "FaucetTask"),
    "box": ("scout.envs.metaworld.tasks.box", "BoxTask"),
}
TASK_NAMES = tuple(_MODULES)


def get_task(name: str) -> MetaWorldTask:
    if name not in _MODULES:
        raise KeyError(f"Unknown Meta-World task '{name}'. Available: {TASK_NAMES}")
    module, cls = _MODULES[name]
    return getattr(importlib.import_module(module), cls)()


__all__ = ["MetaWorldTask", "get_task", "TASK_NAMES"]
