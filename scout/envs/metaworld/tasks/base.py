"""Base class describing how one ISE task is collected and rolled out.

The collection loop (:mod:`scout.envs.metaworld.collect`) and the rollout loop
(:mod:`scout.envs.metaworld.rollout`) are shared by all tasks; everything that
differs between tasks is a hook on this class:

* which MuJoCo bodies form the object cloud, and whether to grey-filter it
* which gym env(s) to build and how to configure them
* the scripted expert used for data collection
* the evaluation policy that executes a target predicted by SCOUT
* how to reset the object between attempts
* small per-env quirks (termination handling, canonical observations)
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from scout.envs.metaworld.pointcloud import get_point_cloud


class MetaWorldTask:
    name: str = ""
    env_names: Sequence[str] = ()
    object_bodies: Sequence[str] = ()
    grey_filter: bool = False
    # Visualisation: which two world axes to plot and which to histogram.
    viz_axes: Tuple[int, int] = (0, 2)
    viz_bin_axis: int = 0
    viz_axis_labels: Tuple[str, str] = ("x", "z  (height)")

    # ── environment construction ─────────────────────────────────────────────
    def gym_kwargs(self, env_name: str, for_rollout: bool) -> dict:
        return {}

    def make_env(self, env_name: str, for_rollout: bool = False):
        env = gym.make("Meta-World/MT1", env_name=env_name, render_mode="rgb_array",
                       **self.gym_kwargs(env_name, for_rollout))
        return env, env.unwrapped

    def after_reset(self, base_env, max_steps: int, for_rollout: bool):
        """Called once after ``env.reset()`` (e.g. to lift path-length limits)."""

    def choose_env_name(self, dataset: List[dict]) -> str:
        """Env for the next collected episode (override to balance variants)."""
        return str(np.random.choice(list(self.env_names)))

    # ── observation ──────────────────────────────────────────────────────────
    def object_cloud(self, env, renderer):
        _, _, pts, clr = get_point_cloud(env, renderer, self.object_bodies, self.grey_filter)
        return pts, clr

    def canonical_observation(self, env, base_env, renderer) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Optional per-episode observation reused as the query cloud for every
        attempt (faucet).  Return ``None`` to render fresh each attempt."""
        return None

    # ── scripted expert (data collection) ────────────────────────────────────
    def make_expert_policy(self, env_name: str, base_env):
        raise NotImplementedError

    def collection_may_continue(self, policy) -> bool:
        """Extra stop condition for the collection loop (push bar caps attempts)."""
        return True

    def stop_on_termination(self) -> bool:
        """Whether gym ``terminated``/``truncated`` ends a collection episode."""
        return False

    # ── evaluation policy (rollout) ──────────────────────────────────────────
    def make_eval_policy(self, env_name: str, base_env):
        """Policy object reused across attempts, or ``None`` for a fresh one each attempt."""
        raise NotImplementedError

    def begin_attempt(self, policy, env_name: str, base_env, target: np.ndarray):
        """Prime the policy with the predicted target; return the policy to run."""
        policy.reset_attempt(target)
        return policy

    def phase_name(self, policy) -> str:
        return ["hover", "descend", "execute"][min(getattr(policy, "sub_phase", 0), 2)]

    def gt_target(self, obs, base_env) -> Optional[np.ndarray]:
        """Ground-truth target for visualisation only."""
        return None

    def handle_termination(self, terminated: bool, truncated: bool) -> Tuple[bool, bool]:
        """Return ``(break_attempt, reset_env)`` for a gym termination signal."""
        return (terminated or truncated), True

    # ── object reset between attempts ────────────────────────────────────────
    def reset_object(self, base_env, env_name: str):
        raise NotImplementedError


def reset_free_object(base_env):
    """Shared reset for the free-joint bar objects (pick / push bar).

    qpos layout: 9-DOF arm + 7-DOF free joint (3 pos + 4 quat)."""
    base_env._set_obj_xyz(base_env.obj_init_pos)
    qpos = base_env.data.qpos.flat.copy()
    qvel = base_env.data.qvel.flat.copy()
    qpos[12:16] = [1.0, 0.0, 0.0, 0.0]
    qvel[9:15] = 0.0
    base_env.set_state(qpos, qvel)
