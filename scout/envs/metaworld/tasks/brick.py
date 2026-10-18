"""Slide Brick: tilt the brick to an apex height and let it slide back into a
narrow rest window; the right apex depends on the hidden ramp friction.

Action = 3-D lift apex ``(brick_x0, brick_y0, apex_z)``; only ``apex_z``
carries the friction signal.  Object cloud = the brick body (geom 40) seen from
the ``corner`` camera at 640x480.

**Needs MuJoCo <= 2.3.7** (the ``scout-brick`` environment, see the README).
The released checkpoint was trained in the original mujoco-py Meta-World; from
MuJoCo 3.0 the reset and slide dynamics change and the expert's own
ground-truth apex succeeds 7% of the time instead of 90%, so :class:`BrickTask`
refuses to build the env under MuJoCo 3.

Point clouds: segmentation rendering is broken in MuJoCo 2.3.7 on NVIDIA EGL,
so the brick mask is computed from depth instead (pixels where the brick-only
depth matches the full-scene depth, i.e. the visible part of the brick), then
cleaned with the same 3x3 median filter as the original pipeline.
"""

from __future__ import annotations

import cv2
import mujoco
import numpy as np
from metaworld.envs.sawyer_brick_slide_ise_v2 import BRICK_GEOM, FRICTION_RANGE
from metaworld.policies.sawyer_brick_slide_ise_policy import SawyerBrickSlideIsePolicyAdaptive

from scout.envs.metaworld.pointcloud import close_renderer
from scout.envs.metaworld.tasks.base import MetaWorldTask

CAMERA = "corner"
WIDTH, HEIGHT = 640, 480
# Geoms are hidden from a render by moving them to group 3, which the default
# visualisation options do not draw.  (A custom MjvOption.geomgroup does not
# hide every group under MuJoCo 2.3.7.)
_HIDDEN_GROUP = 3


def check_mujoco_version():
    if int(mujoco.__version__.split(".")[0]) >= 3:
        raise RuntimeError(
            f"The brick task needs MuJoCo <= 2.3.7, found {mujoco.__version__}. From MuJoCo 3.0 the "
            "brick-slide dynamics change and the released checkpoint (and the scripted expert) no "
            "longer work. Use the scout-brick environment (README, 'Slide Brick').")


# ── observation ──────────────────────────────────────────────────────────────
def _depth(renderer, data):
    renderer.update_scene(data, camera=CAMERA)
    return renderer.render().copy()


def brick_point_cloud(base_env, renderer) -> np.ndarray:
    """World-frame points on the visible surface of the brick body geom."""
    model, data = base_env.model, base_env.data
    groups = model.geom_group.copy()        # geom groups only affect rendering, not physics
    renderer.enable_depth_rendering()
    try:
        depth_all = _depth(renderer, data)
        model.geom_group[:] = _HIDDEN_GROUP
        depth_bg = _depth(renderer, data)                   # no geoms
        model.geom_group[BRICK_GEOM] = groups[BRICK_GEOM]
        depth_brick = _depth(renderer, data)                # brick only
    finally:
        model.geom_group[:] = groups
        renderer.disable_depth_rendering()

    mask = (depth_brick < depth_bg - 1e-4) & (np.abs(depth_brick - depth_all) < 1e-4)
    mask = cv2.medianBlur(mask.astype(np.uint8) * 255, 3) > 0
    v, u = np.nonzero(mask)
    z = depth_all[v, u].astype(np.float64)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA)
    f = 0.5 * HEIGHT / np.tan(np.deg2rad(model.cam_fovy[cam_id]) / 2)
    # MuJoCo cameras look along -z with +y up; image rows grow downwards.
    p_cam = np.stack([(u - (WIDTH - 1) / 2.0) * z / f,
                      -(v - (HEIGHT - 1) / 2.0) * z / f,
                      -z], axis=1)
    R = data.cam_xmat[cam_id].reshape(3, 3)
    return (p_cam @ R.T + data.cam_xpos[cam_id]).astype(np.float32)


# ── policies ─────────────────────────────────────────────────────────────────
class ExpertBrickPolicy(SawyerBrickSlideIsePolicyAdaptive):
    """Scripted expert, exposing its per-attempt apex as ``target`` for the collector."""

    @property
    def target(self):
        return self.target_lift_apex


class EvalBrickPolicy(SawyerBrickSlideIsePolicyAdaptive):
    """Adaptive policy driven by a model-predicted lift apex (``target[2]``).

    Same tilt-up -> retreat-and-wait motion as the expert; the apex height
    comes from the model instead of the friction lookup.
    """

    def __init__(self, p_friction=0.3):
        super().__init__(p_friction=p_friction)
        self._model_target = None

    def _init_adaptive(self, brick_xy, hand_pos):
        # Bypass wrong-apex sampling; still record the brick / hand start for the retreat.
        self.adaptive_n = 999
        self.wrong_highests = [0.0] * 999
        self.brick_init_xy = np.asarray(brick_xy, dtype=np.float32).copy()
        self.init_hand_pos = np.asarray(hand_pos, dtype=np.float32).copy()

    def reset_attempt(self, target):
        self._model_target = np.asarray(target, dtype=np.float32)
        self.sub_phase = 0
        self.lift_steps = 0
        self.still_count = 0
        self.attempt_done = False
        self.last_pos = None
        self.peak_z = -np.inf
        self.stall_steps = 0

    def _adaptive_desired_pos(self, o_d):
        pos_brick = o_d['lid_pos']
        brick_z = pos_brick[2]

        self.current_is_keyframe = False
        self.keyframe_type = None

        self.target_highest = (float(self._model_target[2]) if self._model_target is not None
                               else float(self.highest))
        self.target_lift_apex = (self._model_target.copy() if self._model_target is not None
                                 else np.array([pos_brick[0], pos_brick[1], self.target_highest],
                                               dtype=np.float32))

        # sub_phase 0: tilt up (with plateau detector)
        if self.sub_phase == 0:
            if self.lift_steps == 0:
                self.current_is_keyframe = True
                self.keyframe_type = 'hover'
                self.peak_z = brick_z
                self.stall_steps = 0

            if brick_z > self.peak_z + 5e-4:
                self.peak_z = brick_z
                self.stall_steps = 0
            else:
                self.stall_steps += 1

            if brick_z > self.target_highest:
                self.sub_phase = 1
            elif self.lift_steps >= 30 and self.stall_steps >= 15:
                self.sub_phase = 1
            elif self.lift_steps > 250:
                self.sub_phase = 1
            else:
                self.lift_steps += 1
                return pos_brick + np.array([0.2, 0.0, -0.1])

        # sub_phase 1: retreat and wait for the brick to settle
        if self.sub_phase == 1:
            delta = (float(np.linalg.norm(pos_brick - self.last_pos))
                     if self.last_pos is not None else 1.0)
            if delta < 1e-4:
                self.still_count += 1
            else:
                self.still_count = 0
            self.last_pos = pos_brick.copy()

            if self.still_count >= 10:
                self.current_is_keyframe = True
                self.keyframe_type = 'lift'
                self.attempt_done = True
                self.sub_phase = 0
                self.lift_steps = 0
                self.last_pos = None
                self.still_count = 0
                self.peak_z = -np.inf
                self.stall_steps = 0
            return np.array([self.brick_init_xy[0] - 0.4, self.brick_init_xy[1], 0.45],
                            dtype=np.float32)


# ── task ─────────────────────────────────────────────────────────────────────
class BrickTask(MetaWorldTask):
    name = "brick"
    env_names = ("brick-slide-v2",)
    object_bodies = ("top_link",)
    viz_axes = (0, 2)
    viz_bin_axis = 2
    viz_axis_labels = ("x  (slide direction)", "z  (height)")

    def gym_kwargs(self, env_name, for_rollout):
        return {"max_episode_steps": None}

    def make_env(self, env_name, for_rollout=False):
        check_mujoco_version()
        env, base_env = super().make_env(env_name, for_rollout)
        base_env.set_friction(np.random.uniform(*FRICTION_RANGE))
        return env, base_env

    def after_reset(self, base_env, max_steps, for_rollout):
        # Arm + mocap pose right after reset, restored before every attempt.
        base_env._brick_init_state = {
            "sawyer_qpos": base_env.data.qpos.flat[0:9].copy(),
            "mocap_pos": base_env.data.mocap_pos.copy(),
            "mocap_quat": base_env.data.mocap_quat.copy(),
        }

    _renderer = None            # (model, renderer) of the current episode's env

    def object_cloud(self, env, renderer):
        base_env = env.unwrapped
        if self._renderer is None or self._renderer[0] is not base_env.model:
            if self._renderer is not None:
                close_renderer(self._renderer[1])     # previous episode's env
            self._renderer = (base_env.model, mujoco.Renderer(base_env.model, HEIGHT, WIDTH))
        pts = brick_point_cloud(base_env, self._renderer[1])
        return pts, np.zeros_like(pts)

    def make_expert_policy(self, env_name, base_env):
        return ExpertBrickPolicy(p_friction=base_env.friction)

    def collection_may_continue(self, policy) -> bool:
        # The expert's last planned attempt is the correct one; stop once it is used up.
        return policy.adaptive_n is None or policy.adaptive_phase <= policy.adaptive_n

    def counts_as_success(self, policy) -> bool:
        # Only the planned (correct) attempt may end a collected episode; a wrong
        # apex that happens to succeed would otherwise be stored as ground truth.
        return policy.adaptive_phase == policy.adaptive_n

    def make_eval_policy(self, env_name, base_env):
        return EvalBrickPolicy(p_friction=base_env.friction)

    def phase_name(self, policy):
        return ["tilt up", "retreat / settle"][min(policy.sub_phase, 1)]

    def gt_target(self, obs, base_env):
        return SawyerBrickSlideIsePolicyAdaptive(p_friction=base_env.friction).get_gt_action(obs)

    def reset_object(self, base_env, env_name):
        """Brick back to its rest pose; arm and mocap back to the post-reset pose."""
        init = base_env._brick_init_state
        qpos = base_env.data.qpos.flat.copy()
        qvel = base_env.data.qvel.flat.copy()
        qpos[0:9] = init["sawyer_qpos"]
        qvel[0:9] = 0.0
        qpos[9:12] = base_env.brick_init_pos()
        qpos[12:16] = [1.0, 0.0, 0.0, 0.0]
        qvel[9:15] = 0.0
        base_env.set_state(qpos, qvel)
        base_env.data.mocap_pos[:] = init["mocap_pos"]
        base_env.data.mocap_quat[:] = init["mocap_quat"]
        base_env.threshold = False
        base_env.last_height = 0.0
        base_env.curr_path_length = 0
        return base_env._get_obs()
