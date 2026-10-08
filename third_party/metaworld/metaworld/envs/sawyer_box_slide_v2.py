import numpy as np
from gymnasium.spaces import Box

from metaworld.utils import reward_utils
from metaworld.asset_path_utils import full_v2_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv, _assert_task_is_set


class SawyerBoxSlideEnvV2(SawyerXYZEnv):

    def __init__(self, render_mode = None, **kwargs):

        hand_low = (-0.5, 0.40, 0.1)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.10, 0.7, 0.02)
        obj_high = (0.10, 0.75, 0.02)
        # goal_low = (-0.1, 0.7, 0.133)
        # goal_high = (0.1, 0.8, 0.133)

        super().__init__(
            # self.model_name,
            render_mode=render_mode,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': .3,
            'obj_init_pos': np.array([0, 0.55, 0.02], dtype=np.float32),
            'hand_init_pos': np.array((0, 0.6, 0.2), dtype=np.float32),
        }
        self.goal = np.array([0.0, 0.75, 0.133])
        goal_low = self.goal.copy()
        goal_high = self.goal.copy()
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle'] 
        self.hand_init_pos = self.init_config['hand_init_pos']

        self._target_to_obj_init = None

        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        # self._random_reset_space = Box(
        #     np.hstack((obj_low, goal_low)),
        #     np.hstack((obj_high, goal_high)),
        # )
        self._random_reset_space = Box(
            np.hstack((obj_low)),
            np.hstack((obj_high)),
        )

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_box_slide.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            reward_grab,
            reward_ready,
            reward_success,
            success
        ) = self.compute_reward(action, obs)

        info = {
            'success': float(success),
            'near_object': reward_ready,
            'grasp_success': reward_grab >= 0.5,
            'grasp_reward': reward_grab,
            'in_place_reward': reward_success,
            'obj_to_target': 0,
            'unscaled_reward': reward,
        }

        return reward, info

    @property
    def _target_site_config(self):
        return []

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id('BoxHandleGeom')

    def _get_pos_objects(self):
        return self.get_body_com('top_link')

    def _get_quat_objects(self):
        return self.data.body("top_link").xquat
    
    def _set_obj_xyz(self, pos):
        """Override the base method to handle the 1D slide joint instead of a 7D freejoint."""
        
        # 1. Move the root body of the lid to the randomized XYZ position
        self.model.body("top_link").pos = pos.copy()
        
        # 2. Reset the slide joint's state to 0 (fully closed)
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        
        # The Sawyer arm has 9 joints (indices 0-8). 
        # Our new slide joint is at index 9.
        qpos[9] = 0.0
        qvel[9] = 0.0
        
        self.set_state(qpos, qvel)

    def reset_model(self):
        # print(self._get_state_rand_vec())
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        box_height = self.get_body_com('boxbody')[2]

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            # while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.25:
            #     goal_pos = self._get_state_rand_vec()
            self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]])).copy()
            self._target_pos = goal_pos[-3:].copy()

        self.model.body("boxbody").pos = np.concatenate(
            [self.obj_init_pos[:2], [box_height]]
        )
        self._set_obj_xyz(self.obj_init_pos)
        self._target_pos[1] -= 0.20

        return self._get_obs()

    @staticmethod
    def _reward_grab_effort(actions):
        return (np.clip(actions[3], -1, 1) + 1.0) / 2.0

    @staticmethod
    def _reward_quat(obs):
        # Ideal upright lid has quat [.707, 0, 0, .707]
        # Rather than deal with an angle between quaternions, just approximate:
        ideal = np.array([0.707, 0, 0, -0.707])
        # print(obs[7:11], ideal)
        error = np.linalg.norm(obs[7:11] - ideal)
        return max(1.0 - error/0.2, 0.0)

    @staticmethod
    def _reward_pos(obs, target_pos):
        hand = obs[:3]
        lid = obs[4:7] + np.array([.0, .0, .02])

        threshold = 0.03 # 0.02
        # floor is a 3D funnel centered on the lid's handle
        radius = np.linalg.norm(hand[:2] - lid[:2])
        if radius <= threshold:
            floor = 0.0
        else:
            floor = 0.04 * np.log(radius - threshold) + 0.4
        # prevent the hand from running into the handle prematurely by keeping
        # it above the "floor"
        
        above_floor = 1.0 if hand[2] >= floor else reward_utils.tolerance(
            floor - hand[2],
            bounds=(0.0, 0.01),
            margin=floor / 2.0,
            sigmoid='long_tail',
        )
        # grab the lid's handle
        in_place = reward_utils.tolerance(
            np.linalg.norm(hand - lid),
            bounds=(0, 0.02),
            margin=0.5,
            sigmoid='long_tail',
        )
        # print(radius, hand[2], floor, above_floor, in_place)
        ready_to_lift = reward_utils.hamacher_product(above_floor, in_place)

        # now actually put the lid on the box
        pos_error = target_pos - lid
        error_scale = np.array([1., 1., 3.])  # Emphasize Z error
        a = 0.2  # Relative importance of just *trying* to lift the lid at all
        b = 0.8  # Relative importance of placing the lid on the box
        lifted = a * float(lid[2] > 0.04) + b * reward_utils.tolerance(
            np.linalg.norm(pos_error * error_scale),
            bounds=(0, 0.05),
            margin=0.25,
            sigmoid='long_tail',
        )

        return ready_to_lift, lifted
    
    # @staticmethod
    # def _reward_pos(obs, target_pos):
    #     hand = obs[:3]
    #     # The center of the handle
    #     handle = obs[4:7] + np.array([.0, .0, .02])

    #     # ==========================================
    #     # PHASE 1: Smooth Approach (Hover then Drop)
    #     # ==========================================
    #     # Calculate horizontal distance to the handle
    #     xy_dist = np.linalg.norm(hand[:2] - handle[:2])

    #     # Create a smooth "Hover Factor" from 0.0 to 1.0.
    #     # If the hand is 5cm or further away, the factor is 1.0.
    #     # As the hand moves perfectly over the handle, it smoothly drops to 0.0.
    #     hover_factor = np.clip(xy_dist / 0.05, 0.0, 0.5)
        
    #     # The target Z height smoothly drops from 15cm above the handle down to the handle itself
    #     target_z = handle[2] + (0.15 * hover_factor)

    #     # The robot's goal is simply to reach this dynamic point in 3D space
    #     dynamic_target = np.array([handle[0], handle[1], target_z])
    #     dist_to_target = np.linalg.norm(hand - dynamic_target)

    #     # A smooth, bell-shaped curve that peaks at 1.0 when the hand catches the dynamic target
    #     ready_to_lift = reward_utils.tolerance(
    #         dist_to_target,
    #         bounds=(0, 0.02),
    #         margin=0.5,
    #         sigmoid='long_tail',
    #     )

    #     # ==========================================
    #     # PHASE 2: Smooth Execution (The Slide)
    #     # ==========================================
    #     # Since the slide joint locks the lid's Z height, we only calculate X/Y error
    #     pos_error = np.linalg.norm(target_pos[:2] - handle[:2])

    #     # A smooth curve that rewards pushing the lid to the goal
    #     pushed = reward_utils.tolerance(
    #         pos_error,
    #         bounds=(0, 0.02),
    #         margin=0.25,
    #         sigmoid='long_tail',
    #     )

    #     # Give a small bonus for touching the handle, but scale the bulk of 
    #     # the reward based on actually pushing the lid.
    #     return ready_to_lift, pushed

    def compute_reward(self, actions, obs):
        reward_grab = SawyerBoxSlideEnvV2._reward_grab_effort(actions)
        reward_quat = SawyerBoxSlideEnvV2._reward_quat(obs)
        reward_steps = SawyerBoxSlideEnvV2._reward_pos(obs, self._target_pos)

        reward = sum((
            2.0 * reward_utils.hamacher_product(reward_grab, reward_steps[0]),
            8.0 * reward_steps[1],
        ))

        # Override reward on success
        success = np.linalg.norm(obs[4:7] - self._target_pos) < 0.08
        if success:
            reward = 10.0

        # STRONG emphasis on proper lid orientation to prevent reward hacking
        # (otherwise agent learns to kick-flip the lid onto the box)
        # print(reward, reward_quat)

        reward *= reward_quat
        
        return (
            reward,
            reward_grab,
            *reward_steps,
            success,
        )
