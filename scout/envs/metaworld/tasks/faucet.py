"""Turn Faucet: binary task (open vs close); action = 3-D push goal.

The two env variants differ only in invisible scene details (an invisible
wall and goal sites).  Those would leak the variant into the point cloud, so
the *query* observation is rendered from a canonical scene with the leak
sources moved away; the *outcome* clouds are rendered from the real scene so
they carry the information TTT needs.
"""

import mujoco
import numpy as np
from metaworld.policies.sawyer_faucet_close_v3_policy import SawyerFaucetCloseV3Policy
from metaworld.policies.sawyer_faucet_open_v3_policy import SawyerFaucetOpenV3Policy

from scout.envs.metaworld.pointcloud import get_point_cloud
from scout.envs.metaworld.tasks.base import MetaWorldTask

_CANONICAL_FAUCET_BASE = np.array([0.0, 0.825, 0.0], dtype=np.float64)
_FAR_AWAY = np.array([0.0, 0.0, -10.0], dtype=np.float64)
OBJECT_BODIES = ("faucet_link", "faucet_link2")   # spout + handle; faucetBase excluded


def _set_hinge_zero(base_env):
    qpos = base_env.data.qpos.flat.copy()
    qvel = base_env.data.qvel.flat.copy()
    qpos[base_env.model.joint("knob_Joint_1").qposadr[0]] = 0.0
    qvel[base_env.model.joint("knob_Joint_1").dofadr[0]] = 0.0
    base_env.set_state(qpos, qvel)


def reset_faucet(base_env):
    """Reset the handle hinge to zero and return the arm to its init pose."""
    _set_hinge_zero(base_env)
    base_env._reset_hand()


def capture_canonical_obj_pcd(env, base_env, renderer):
    """Render the object cloud with the three leak channels neutralised:

    1. ``box_left`` wall (different x per variant)       -> moved far away
    2. ``goal_open`` / ``goal_close`` sites (mirrored)    -> moved far away
    3. ``faucetBase`` position (seed dependent)           -> canonical midpoint

    The cloud is translated back into real-world coordinates.  Leaves the env
    in a clean reset state at the real faucet pose; the caller should refresh
    ``obs`` with ``base_env._get_obs()``.
    """
    m = base_env.model
    saved = (m.geom("box_left").pos.copy(), m.site("goal_open").pos.copy(),
             m.site("goal_close").pos.copy(), m.body("faucetBase").pos.copy())

    # Hide leak sources *before* the reset: its settling steps otherwise leave
    # an asymmetric contact state in the close variant.
    m.geom("box_left").pos = _FAR_AWAY
    m.site("goal_open").pos = _FAR_AWAY
    m.site("goal_close").pos = _FAR_AWAY
    m.body("faucetBase").pos = _CANONICAL_FAUCET_BASE
    reset_faucet(base_env)
    mujoco.mj_forward(m, base_env.data)
    _, _, pts, clr = get_point_cloud(env, renderer, OBJECT_BODIES)
    pts_real = pts + (saved[3] - _CANONICAL_FAUCET_BASE).astype(np.float32)[None, :]

    m.geom("box_left").pos, m.site("goal_open").pos, m.site("goal_close").pos, m.body("faucetBase").pos = saved
    # Restore hinge = 0 *without* re-running the hand reset (its settling
    # would rotate the knob against the wall in the close variant and make
    # obs disagree with the canonical cloud).
    _set_hinge_zero(base_env)
    mujoco.mj_forward(m, base_env.data)
    return pts_real, clr


class _EvalFaucetMixin:
    """Inject an external push goal and lock the push direction to the
    *initial* handle pose (the handle drifts during the approach in the close
    variant, which would otherwise flip wrong predictions into successes)."""

    def reset_attempt(self, target):
        self.external_target = np.array(target, dtype=np.float32)
        self.sub_phase = 0
        self.exec_steps = 0
        self.stuck_steps = 0
        self.release_countdown = 0
        self.attempt_done = False
        self.current_is_keyframe = False
        self.keyframe_type = None
        self.prev_faucet_pos = None
        self._initial_handle_tip = None

    def _adaptive_desired_pos(self, o_d):
        if self._initial_handle_tip is None:
            self._initial_handle_tip = o_d["faucet_pos"].copy()
        prev_phase = self.sub_phase
        result = super()._adaptive_desired_pos(o_d)
        if prev_phase == 1 and self.sub_phase == 2:
            push_goal = self.external_target if self.external_target is not None else self.target
            self.push_right = bool(push_goal[0] > self._initial_handle_tip[0])
        return result


class EvalFaucetOpenPolicy(_EvalFaucetMixin, SawyerFaucetOpenV3Policy):
    pass


class EvalFaucetClosePolicy(_EvalFaucetMixin, SawyerFaucetCloseV3Policy):
    pass


class FaucetTask(MetaWorldTask):
    name = "faucet"
    env_names = ("faucet-open-v3", "faucet-close-v3")
    object_bodies = OBJECT_BODIES
    grey_filter = False
    viz_axes = (0, 2)
    viz_bin_axis = 0
    viz_axis_labels = ("x  (push direction)", "z  (height)")

    def gym_kwargs(self, env_name, for_rollout):
        return {} if for_rollout else {"max_episode_steps": 2000}

    def after_reset(self, base_env, max_steps, for_rollout):
        if not for_rollout:
            base_env.max_path_length = max_steps

    def choose_env_name(self, dataset):
        n_open = sum(1 for s in dataset if s["env_name"] == "faucet-open-v3")
        n_close = len(dataset) - n_open
        if n_open < n_close:
            return "faucet-open-v3"
        if n_close < n_open:
            return "faucet-close-v3"
        return str(np.random.choice(list(self.env_names)))

    def canonical_observation(self, env, base_env, renderer):
        return capture_canonical_obj_pcd(env, base_env, renderer)

    def make_expert_policy(self, env_name, base_env):
        return SawyerFaucetOpenV3Policy() if env_name == "faucet-open-v3" else SawyerFaucetCloseV3Policy()

    def make_eval_policy(self, env_name, base_env):
        return EvalFaucetOpenPolicy() if env_name == "faucet-open-v3" else EvalFaucetClosePolicy()

    def reset_object(self, base_env, env_name):
        reset_faucet(base_env)
        return base_env._get_obs()
