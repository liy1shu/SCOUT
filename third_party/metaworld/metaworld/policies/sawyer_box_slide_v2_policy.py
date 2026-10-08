import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerBoxSlideV2Policy(Policy):

    def __init__(self):
        # Legacy state for get_action path
        self.checkpoint = False
        self.mode = np.random.choice(['pull', 'slide'])
        self.stuck_steps = 0
        self.stuck_threshold = 3
        self.prev_hand_pos = None

        # Adaptive state (mirrors SawyerPickBarV2Policy naming)
        self.right_strategy    = 'slide'
        self.wrong_strategy    = 'pull'
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

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=10.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    def _desired_pos(self, o_d):
        pos_curr = o_d['hand_pos']
        pos_handle = o_d['lid_pos'] + np.array([.0, .0, .1])
        pos_goal = pos_handle + np.array([0.0, -0.25, 0.0])

        if not self.checkpoint and np.linalg.norm(pos_curr[:2] - pos_handle[:2]) > 0.02:
            return pos_handle + np.array([0., 0., 0.15])
        elif not self.checkpoint and abs(pos_curr[2] - pos_handle[2]) > 0.02:
            return pos_handle
        else:
            self.checkpoint = True
            return pos_goal

    def _grab_effort(self, o_d):
        pos_curr = o_d['hand_pos']
        pos_handle = o_d['lid_pos'] + np.array([.0, .0, .02])

        if np.linalg.norm(pos_curr - pos_handle) < 0.025 or self.checkpoint:
            return 1.0
        else:
            return -1.0

    def get_adaptive_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({'delta_pos': np.arange(3), 'grab_effort': 3})
        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._adaptive_desired_pos(o_d), p=10.)
        action['grab_effort'] = self._adaptive_grab_effort(o_d)
        return action.array

    def _adaptive_desired_pos(self, o_d):
        pos_curr = o_d['hand_pos']
        pos_lid  = o_d['lid_pos'] + np.array([.0, .0, .10])

        self.current_is_keyframe = False
        self.keyframe_type = None

        # sub_phase 0: hover above handle; compute preview target for hover keyframe capture
        if self.sub_phase == 0:
            if self.external_target is not None:
                self.target = np.asarray(self.external_target, dtype=np.float32)
            elif self.current_strategy == 'slide':
                self.target = pos_lid + np.array([0., -0.25, 0.])
            else:
                self.target = pos_lid + np.array([0., 0., 0.30])

            if np.linalg.norm(pos_curr[:2] - pos_lid[:2]) > 0.02:
                return pos_lid + np.array([0., 0., 0.15])
            else:
                self.current_is_keyframe = True
                self.keyframe_type = 'hover'
                self.sub_phase = 1

        # sub_phase 1: descend to handle
        if self.sub_phase == 1:
            if abs(pos_curr[2] - pos_lid[2]) > 0.04:
                return pos_lid
            else:
                # Cache fixed execution target from current handle position
                if self.external_target is not None:
                    self.target = np.asarray(self.external_target, dtype=np.float32)
                elif self.current_strategy == 'slide':
                    self.target = pos_lid + np.array([0., -0.25, 0.])
                else:
                    self.target = pos_lid + np.array([0., 0., 0.30])
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
                        'pull' if self.current_strategy == 'slide' else 'slide'
                    )
                self.sub_phase = 0
            return self.target

    def _adaptive_grab_effort(self, o_d):
        if self.release_countdown > 0 or self.attempt_done:
            return -1.
        if self.sub_phase == 2:
            if self.current_strategy == 'pull':
                return -1.  # open gripper during wrong strategy so lid doesn't slide
            return 1.
        pos_curr = o_d['hand_pos']
        pos_lid  = o_d['lid_pos'] + np.array([.0, .0, .10])
        if np.linalg.norm(pos_curr[:2] - pos_lid[:2]) > 0.02 \
                or abs(pos_curr[2] - pos_lid[2]) > 0.12:
            return 0.
        return 1.
