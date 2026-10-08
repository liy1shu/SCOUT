"""Push Bar: push a bar with a hidden centre of mass; action = 3-D contact point."""

import numpy as np
from metaworld.policies.sawyer_push_test_v2_policy import SawyerPushTestV2Policy

from scout.envs.metaworld.tasks.base import MetaWorldTask, reset_free_object


class PushBarTask(MetaWorldTask):
    name = "push_bar"
    env_names = ("push-test-v2",)
    object_bodies = ("obj",)
    grey_filter = True
    viz_axes = (0, 2)
    viz_bin_axis = 0
    viz_axis_labels = ("x  (along rod)", "z  (height)")

    COLLECT_MAX_ATTEMPTS = 20

    def gym_kwargs(self, env_name, for_rollout):
        return {"max_episode_steps": None}

    def after_reset(self, base_env, max_steps, for_rollout):
        base_env.max_path_length = 2000 if for_rollout else 100000

    def make_expert_policy(self, env_name, base_env):
        return SawyerPushTestV2Policy(env_cm_offset=base_env.cm_offset)

    def collection_may_continue(self, policy):
        return policy.adaptive_phase < self.COLLECT_MAX_ATTEMPTS

    # A fresh policy per attempt; ``external_target`` drives it and guarantees
    # the full push distance regardless of the adaptive phase.
    def make_eval_policy(self, env_name, base_env):
        return None

    def begin_attempt(self, policy, env_name, base_env, target):
        policy = SawyerPushTestV2Policy(env_cm_offset=base_env.cm_offset)
        policy.adaptive_n = 0
        policy.wrong_offsets = []
        policy.external_target = np.asarray(target, dtype=np.float32)
        return policy

    def phase_name(self, policy):
        return ["hover", "descend", "push"][min(policy.sub_phase, 2)]

    def gt_target(self, obs, base_env):
        return np.asarray(obs[4:7]).copy()          # true rod centre of mass

    def handle_termination(self, terminated, truncated):
        return terminated, False

    def reset_object(self, base_env, env_name):
        reset_free_object(base_env)
        base_env.curr_path_length = 0
        return base_env._get_obs()
