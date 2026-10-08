"""Meta-World (MuJoCo) tasks.

Importing this package sets ``MUJOCO_GL=egl`` for headless rendering and
caps the CUDA allocator split size (PyTorch 2.1.x + ``expandable_segments``
asserts inside gradient-checkpointed double backward).  Both must happen
before ``mujoco`` / ``torch`` are imported.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

from scout.envs.metaworld.tasks import get_task, TASK_NAMES  # noqa: E402

__all__ = ["get_task", "TASK_NAMES"]
