"""Pick Bar: lift a bar whose centre of mass is hidden; action = 3-D grasp point."""

import numpy as np
from metaworld.policies.sawyer_pick_bar_v2_policy import SawyerPickBarV2Policy

from scout.envs.metaworld.tasks.base import MetaWorldTask, reset_free_object


class EvalBarPolicy(SawyerPickBarV2Policy):
    """Expert state machine driven by an externally supplied grasp point."""

    def __init__(self, env_cm_offset=None):
        super().__init__(env_cm_offset=env_cm_offset)
        self._model_target = None
        self.adaptive_n = 999
        self.wrong_offsets = [0.0] * 999

    def reset_attempt(self, target):
        self._model_target = np.array(target, dtype=np.float32)
        self.sub_phase = 0
        self.release_countdown = 0
        self.attempt_done = False

    def _adaptive_desired_pos(self, o_d):
        pos_curr = o_d["hand_pos"]
        pos_goal = o_d["goal_pos"]
        self.current_is_keyframe = False
        self.keyframe_type = None
        self.target = (self._model_target.copy() if self._model_target is not None
                       else o_d["puck_pos"].copy())

        if self.sub_phase == 0:                       # hover above the target
            if np.linalg.norm(pos_curr[:2] - self.target[:2]) > 0.02:
                return [self.target[0], self.target[1], pos_goal[2]]
            self.current_is_keyframe = True
            self.keyframe_type = "hover"
            self.sub_phase = 1

        if self.sub_phase == 1:                       # descend
            if abs(pos_curr[2] - self.target[2]) > 0.04:
                return self.target
            self.sub_phase = 2

        if self.sub_phase == 2:                       # lift, hold, release
            lift_z = pos_goal[2]
            if pos_curr[2] < lift_z - 0.01:
                return [self.target[0], self.target[1], lift_z]
            if self.release_countdown == 0:
                self.current_is_keyframe = True
                self.keyframe_type = "lift"
                self.release_countdown = 25
            self.release_countdown -= 1
            if self.release_countdown == 0:
                self.attempt_done = True
                self.sub_phase = 0
            return [self.target[0], self.target[1], lift_z]

    def _adaptive_grab_effort(self, o_d):
        if self.attempt_done or self.release_countdown > 0:
            return -1.0
        if self.sub_phase == 2:
            return 1.0
        pos_curr = o_d["hand_pos"]
        if (np.linalg.norm(pos_curr[:2] - self.target[:2]) > 0.02
                or abs(pos_curr[2] - self.target[2]) > 0.12):
            return 0.0
        return 0.6


class PickBarTask(MetaWorldTask):
    name = "pick_bar"
    env_names = ("pick-bar-v2",)
    object_bodies = ("obj",)
    grey_filter = True
    viz_axes = (0, 2)
    viz_bin_axis = 0
    viz_axis_labels = ("x  (along bar)", "z  (height)")

    def gym_kwargs(self, env_name, for_rollout):
        return {"max_episode_steps": None} if for_rollout else {}

    def make_expert_policy(self, env_name, base_env):
        return SawyerPickBarV2Policy(env_cm_offset=base_env.cm_offset)

    def make_eval_policy(self, env_name, base_env):
        return EvalBarPolicy(env_cm_offset=base_env.cm_offset)

    def phase_name(self, policy):
        return ["hover", "descend", "lift"][min(policy.sub_phase, 2)]

    def gt_target(self, obs, base_env):
        return SawyerPickBarV2Policy(env_cm_offset=base_env.cm_offset).get_gt_grasp_pos(obs)

    def stop_on_termination(self):
        return True

    def reset_object(self, base_env, env_name):
        reset_free_object(base_env)
        return base_env._get_obs()
