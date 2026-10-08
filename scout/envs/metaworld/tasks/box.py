"""Open Box: binary task (lift the lid vs slide it); action = 3-D post-grasp target."""

import numpy as np
from metaworld.policies.sawyer_box_open_v2_policy import SawyerBoxOpenV2Policy
from metaworld.policies.sawyer_box_slide_v2_policy import SawyerBoxSlideV2Policy

from scout.envs.metaworld.tasks.base import MetaWorldTask


def reset_lid(base_env, env_name):
    base_env._set_obj_xyz(base_env.obj_init_pos)
    qpos = base_env.data.qpos.flat.copy()
    qvel = base_env.data.qvel.flat.copy()
    if env_name == "box-open-v2":          # 9-DOF arm + 7-DOF free joint
        qpos[9:12] = base_env.obj_init_pos
        qpos[12:16] = [1.0, 0.0, 0.0, 0.0]
        qvel[9:15] = 0.0
    else:                                   # box-slide-v2: 9-DOF arm + 1-DOF slide
        qpos[9] = 0.0
        qvel[9] = 0.0
    base_env.set_state(qpos, qvel)


class _EvalBoxMixin:
    def reset_attempt(self, target):
        self.external_target = np.array(target, dtype=np.float32)
        self.sub_phase = 0
        self.exec_steps = 0
        self.release_countdown = 0
        self.attempt_done = False
        self.current_is_keyframe = False
        self.keyframe_type = None


class EvalBoxOpenPolicy(_EvalBoxMixin, SawyerBoxOpenV2Policy):
    pass


class EvalBoxSlidePolicy(_EvalBoxMixin, SawyerBoxSlideV2Policy):
    pass


class BoxTask(MetaWorldTask):
    name = "box"
    env_names = ("box-open-v2", "box-slide-v2")
    object_bodies = ("boxbodytop",)
    grey_filter = False
    # Pull vs slide differ along y (slide direction) and z (height).
    viz_axes = (1, 2)
    viz_bin_axis = 2
    viz_axis_labels = ("y  (slide direction)", "z  (height)")

    def make_expert_policy(self, env_name, base_env):
        return SawyerBoxOpenV2Policy() if env_name == "box-open-v2" else SawyerBoxSlideV2Policy()

    def make_eval_policy(self, env_name, base_env):
        return EvalBoxOpenPolicy() if env_name == "box-open-v2" else EvalBoxSlidePolicy()

    def stop_on_termination(self):
        return True

    def reset_object(self, base_env, env_name):
        reset_lid(base_env, env_name)
        return base_env._get_obs()
