import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerBoxOpenV2Policy(Policy):

    def __init__(self):
        # Legacy state for get_action path
        self.checkpoint = False
        self.mode = np.random.choice(['pull', 'slide'])
        self.stuck_steps = 0
        self.stuck_threshold = 5
        self.prev_lid_pos = None

        # Adaptive state (mirrors SawyerPickBarV2Policy naming)
        self.right_strategy    = 'pull'
        self.wrong_strategy    = 'slide'
        self.current_strategy  = np.random.choice(['pull', 'slide'])
        self.tried_strategies  = []
        self.sub_phase         = 0
        self.exec_steps        = 0
        self.attempt_done      = False
        self.release_countdown = 0
        self.current_is_keyframe = False
        self.keyframe_type     = None
        self.target            = np.zeros(3)
        self.external_target   = None  # if set, overrides strategy-based target in rollout

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'gripper': obs[3],
            'lid_pos': obs[4:7],
            'extra_info_1': obs[7:-3],
            'box_pos': obs[-3:-1],
            'extra_info_2': obs[-1],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=5.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_lid = o_d['lid_pos'] + np.array([.0, .0, +.07])
        pos_box = np.array([*o_d['box_pos'], 0.15]) + np.array([.0, .0, .0])

        if np.linalg.norm(pos_curr[:2] - pos_lid[:2]) > 0.01:
            return np.array([*pos_lid[:2], 0.2])
        elif abs(pos_curr[2] - pos_lid[2]) > 0.05:
            return pos_lid
        elif pos_curr[2] - pos_box[2] < -0.04:
            return np.array([pos_curr[0], pos_curr[1], pos_box[2]])
        else:
            return pos_box + np.array([.0, .0, .3])

    @staticmethod
    def _grab_effort(o_d):
        pos_curr = o_d['hand_pos']
        pos_lid = o_d['lid_pos'] + np.array([.0, .0, +.07])

        if np.linalg.norm(pos_curr[:2] - pos_lid[:2]) > 0.01 or abs(pos_curr[2] - pos_lid[2]) > 0.13:
            return .0
        else:
            return 1.

    def get_adaptive_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({'delta_pos': np.arange(3), 'grab_effort': 3})
        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._adaptive_desired_pos(o_d), p=10.)
        action['grab_effort'] = self._adaptive_grab_effort(o_d)
        return action.array

    def _adaptive_desired_pos(self, o_d):
        pos_curr = o_d['hand_pos']
        pos_lid  = o_d['lid_pos'] + np.array([.0, .0, .07])

        self.current_is_keyframe = False
        self.keyframe_type = None

        # sub_phase 0: hover above handle; compute preview target for hover keyframe capture
        if self.sub_phase == 0:
            if self.external_target is not None:
                self.target = np.asarray(self.external_target, dtype=np.float32)
            elif self.current_strategy == 'pull':
                self.target = np.array([pos_lid[0], pos_lid[1], 0.45])
            else:
                self.target = pos_lid + np.array([0., -0.25, 0.])

            if np.linalg.norm(pos_curr[:2] - pos_lid[:2]) > 0.01:
                return pos_lid + np.array([0., 0., 0.15])
            else:
                self.current_is_keyframe = True
                self.keyframe_type = 'hover'
                self.sub_phase = 1

        # sub_phase 1: descend to handle
        if self.sub_phase == 1:
            if abs(pos_curr[2] - pos_lid[2]) > 0.05:
                return pos_lid
            else:
                # Cache fixed execution target from current handle position
                if self.external_target is not None:
                    self.target = np.asarray(self.external_target, dtype=np.float32)
                elif self.current_strategy == 'pull':
                    self.target = np.array([pos_lid[0], pos_lid[1], 0.45])
                else:
                    self.target = pos_lid + np.array([0., -0.25, 0.])
                self.sub_phase = 2
                self.exec_steps = 0

        # sub_phase 2: execute strategy; timeout triggers attempt_done
        if self.sub_phase == 2:
            dist = np.linalg.norm(pos_curr - self.target)
            if dist > 0.05 and self.exec_steps < 100:
                self.exec_steps += 1
                return self.target

            # Fire 'lift' keyframe once, then hold for physics settling
            if self.release_countdown == 0:
                self.current_is_keyframe = True
                self.keyframe_type = 'lift'
                self.release_countdown = 15
            self.release_countdown -= 1
            if self.release_countdown == 0:
                self.attempt_done = True
                self.tried_strategies.append(self.current_strategy)
                if len(self.tried_strategies) < 2:
                    self.current_strategy = (
                        'slide' if self.current_strategy == 'pull' else 'pull'
                    )
                self.sub_phase = 0
            return self.target

    def _adaptive_grab_effort(self, o_d):
        if self.release_countdown > 0 or self.attempt_done:
            return -1.
        if self.sub_phase == 2:
            return 1.
        pos_curr = o_d['hand_pos']
        pos_lid  = o_d['lid_pos'] + np.array([.0, .0, .07])
        if np.linalg.norm(pos_curr[:2] - pos_lid[:2]) > 0.01 \
                or abs(pos_curr[2] - pos_lid[2]) > 0.13:
            return 0.
        return 1.
