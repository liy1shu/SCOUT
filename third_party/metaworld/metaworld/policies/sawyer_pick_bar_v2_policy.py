# import numpy as np

# from metaworld.policies.action import Action
# from metaworld.policies.policy import Policy, assert_fully_parsed, move


# class SawyerPickBarV2Policy(Policy):

#     def __init__(self, relative_offset=None, **kwargs):
#         if relative_offset is not None:
#             # print(f"POLICY: relative_offset = {relative_offset}")
#             self.relative_offset = relative_offset
#         else:
#             self.relative_offset = 0
#         super().__init__(**kwargs)

#         self.grabbed = False


#     @staticmethod
#     @assert_fully_parsed
#     def _parse_obs(obs):
#         return {
#             'hand_pos': obs[:3],
#             'unused_1': obs[3],
#             'puck_pos': obs[4:7],
#             'unused_2':  obs[7:-3],
#             'goal_pos': obs[-3:],
#         }

#     def get_action(self, obs):
#         o_d = self._parse_obs(obs)

#         action = Action({
#             'delta_pos': np.arange(3),
#             'grab_effort': 3
#         })

#         o_d['puck_pos'] += np.array([-self.relative_offset, 0, 0])

#         action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=10.)
#         action['grab_effort'] = self._grab_effort(o_d)

#         return action.array

#     # @staticmethod
#     def _desired_pos(self, o_d):
#         pos_curr = o_d['hand_pos']
#         pos_puck = o_d['puck_pos'] 
#         pos_goal = o_d['goal_pos']
#         # print(pos_curr, pos_puck, pos_goal)

#         if not self.grabbed and np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.02:
#             return [pos_puck[0], pos_puck[1], pos_goal[2]]
#         elif not self.grabbed and np.linalg.norm(pos_curr[2:] - pos_puck[2:]) > 0.04:
#             return [pos_puck[0], pos_puck[1], pos_puck[2]]
#         else:
#             self.grabbed = True
#             return [pos_puck[0], pos_puck[1], 0.25]
        
#         return pos_goal

#     @staticmethod
#     def _grab_effort(o_d):
#         pos_curr = o_d['hand_pos']
#         pos_puck = o_d['puck_pos']

#         if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.02 or abs(pos_curr[2] - pos_puck[2]) > 0.12:
#             return 0.
#         # While end effector is moving down toward the puck, begin closing the grabber
#         else:
#             return 0.6


import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerPickBarV2Policy(Policy):

    def __init__(self, relative_offset=None, env_cm_offset=None, **kwargs):
        if relative_offset is not None:
            # print(f"POLICY: relative_offset = {relative_offset}")
            self.relative_offset = relative_offset
        else:
            self.relative_offset = 0
        # env_cm_offset is the env's cm_offset (rod center offset from CoM in body frame = 0.9 * cm_offset).
        # Used to compute correct rod bounds for wrong-grasp sampling in get_adaptive_action.
        self.env_pm = 0.9 * env_cm_offset if env_cm_offset is not None else 0.
        super().__init__(**kwargs)

        self.grabbed = False

        # State for get_adaptive_action
        self.adaptive_phase = 0    # current attempt index; >= adaptive_n means success phase
        self.adaptive_n = None     # total failure attempts, drawn lazily on first call
        self.wrong_offsets = None  # list of N wrong grasp offsets
        self.sub_phase = 0         # sub-stage within each attempt (0=hover, 1=descend, 2=lift)
        self.attempt_done = False  # set True when a failure attempt finishes; caller resets bar then clears this
        self.release_countdown = 0  # counts down during release phase (>0 means actively releasing)
        self.current_is_keyframe = False  # set True on the step a keyframe transition occurs
        self.keyframe_type = None         # 'hover' or 'lift' when current_is_keyframe is True


    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_1': obs[3],
            'puck_pos': obs[4:7],
            'unused_2':  obs[7:-3],
            'goal_pos': obs[-3:],
        }
        
    def get_gt_grasp_pos(self, obs):
        return self._parse_obs(obs)['puck_pos']

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        o_d['puck_pos'] += np.array([-self.relative_offset, 0, 0])

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=10.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    # @staticmethod
    def _desired_pos(self, o_d):
        pos_curr = o_d['hand_pos']
        pos_puck = o_d['puck_pos'] 
        pos_goal = o_d['goal_pos']
        # print(pos_curr, pos_puck, pos_goal)

        if not self.grabbed and np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.02:
            return [pos_puck[0], pos_puck[1], pos_goal[2]]
        elif not self.grabbed and np.linalg.norm(pos_curr[2:] - pos_puck[2:]) > 0.04:
            return [pos_puck[0], pos_puck[1], pos_puck[2]]
        else:
            self.grabbed = True
            return [pos_puck[0], pos_puck[1], 0.25]
        
        return pos_goal

    @staticmethod
    def _grab_effort(o_d):
        pos_curr = o_d['hand_pos']
        pos_puck = o_d['puck_pos']

        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.02 or abs(pos_curr[2] - pos_puck[2]) > 0.12:
            return 0.
        # While end effector is moving down toward the puck, begin closing the grabber
        else:
            return 0.6

    def get_adaptive_action(self, obs):
        o_d = self._parse_obs(obs)

        # One-time initialization: draw N and sample wrong offsets
        if self.adaptive_n is None:
            self.adaptive_n = np.random.randint(1, 9)  # uniform in [1, 8]
            # self.adaptive_n = 0
            margin = 0.04   # stay this far from each rod end to avoid slipping off
            # rod center is at +env_pm from CoM; grasp target = puck_pos - off,
            # so off must be in [-env_pm - 0.2 + margin, -env_pm + 0.2 - margin]
            rod_lo, rod_hi = -self.env_pm - 0.2 + margin, -self.env_pm + 0.2 - margin
            offsets = []
            while len(offsets) < self.adaptive_n:
                candidate = np.random.uniform(rod_lo, rod_hi)
                if abs(candidate) >= 0.03:  # must be clearly off CoM
                    offsets.append(candidate)
            self.wrong_offsets = offsets

        action = Action({'delta_pos': np.arange(3), 'grab_effort': 3})
        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._adaptive_desired_pos(o_d), p=10.)
        action['grab_effort'] = self._adaptive_grab_effort(o_d)
        return action.array

    def _adaptive_desired_pos(self, o_d):
        pos_curr = o_d['hand_pos']
        pos_puck = o_d['puck_pos']   # true CoM
        pos_goal = o_d['goal_pos']

        self.current_is_keyframe = False
        self.keyframe_type = None

        is_success = (self.adaptive_phase >= self.adaptive_n)
        if is_success:
            self.target = pos_puck.copy()
        else:
            off = self.wrong_offsets[self.adaptive_phase]
            self.target = pos_puck + np.array([-off, 0., 0.])

        # sub_phase 0: hover above target xy, gripper open
        if self.sub_phase == 0:
            if np.linalg.norm(pos_curr[:2] - self.target[:2]) > 0.02:
                return [self.target[0], self.target[1], pos_goal[2]]
            else:
                self.current_is_keyframe = True
                self.keyframe_type = 'hover'
                self.sub_phase = 1

        # sub_phase 1: descend to puck level
        if self.sub_phase == 1:
            if abs(pos_curr[2] - self.target[2]) > 0.04:
                return self.target
            else:
                self.sub_phase = 2

        # sub_phase 2: lift
        if self.sub_phase == 2:
            lift_z = 0.25 if is_success else 0.20
            if is_success or pos_curr[2] <= 0.18:
                return [self.target[0], self.target[1], lift_z]
            else:
                # Hold position and open gripper for several steps so physics fully releases the bar
                # before signaling the caller to teleport it back to init_pos.
                if self.release_countdown == 0:
                    self.current_is_keyframe = True
                    self.keyframe_type = 'lift'
                    self.release_countdown = 15  # start countdown on first entry
                self.release_countdown -= 1
                if self.release_countdown == 0:
                    self.attempt_done = True
                    self.adaptive_phase += 1
                    self.sub_phase = 0
                return [self.target[0], self.target[1], lift_z]

    def _adaptive_grab_effort(self, o_d):
        if self.attempt_done or self.release_countdown > 0:
            return -1.   # open gripper during release phase and until caller resets bar
        pos_curr = o_d['hand_pos']
        if np.linalg.norm(pos_curr[:2] - self.target[:2]) > 0.02 or abs(pos_curr[2] - self.target[2]) > 0.12:
            return 0.    # stay open until arm is close to target
        else:
            return 0.6   # close only in final approach