"""SCOUT: Scouting the Dynamics Gap — test-time policy adaptation via action-outcome feedback.

Package layout
--------------
scout.configs   dataclass configs + the per-task registry (``get_task_config``)
scout.models    network components and the ``SCOUT`` model bundle
scout.data      datasets (Meta-World attempt histories, ambiguous-door trajectories)
scout.training  meta-training loop and offline evaluation
scout.envs      simulator-specific code: data collection and rollouts
"""

from scout.configs import (
    DataConfig,
    ModelConfig,
    RolloutConfig,
    TaskConfig,
    TrainConfig,
    TASKS,
    get_task_config,
)
from scout.models.scout import SCOUT

__all__ = [
    "SCOUT",
    "DataConfig",
    "ModelConfig",
    "RolloutConfig",
    "TaskConfig",
    "TrainConfig",
    "TASKS",
    "get_task_config",
]
