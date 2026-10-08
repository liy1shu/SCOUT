import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerPushTestV2Policy(Policy):

    def __init__(self, relative_offset=None, env_cm_offset=None, **kwargs):
        if relative_offset is not None:
            self.relative_offset = relative_offset
        else:
            self.relative_offset = 0

        self.env_pm = 0.9 * env_cm_offset if env_cm_offset is not None else 0.

        super().__init__(**kwargs)

        # Adaptive state
        self.adaptive_phase = 0
        self.adaptive_n = None
        self.wrong_offsets = None
        self.sub_phase = 0
        self.attempt_done = False
        self.release_countdown = 0
        self.current_is_keyframe = False
        self.keyframe_type = None
        self.target = np.zeros(3)
        self.push_unit_xy = None
        self.push_start_xy = None
        self.external_target = None

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

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_puck = o_d['puck_pos'] + np.array([-0.01, 0, 0])
        pos_goal = o_d['goal_pos']

        direction = pos_goal[:2] - pos_puck[:2]
        dir_dist = np.linalg.norm(direction)
        direction /= dir_dist

        place = pos_curr[:2] - pos_puck[:2]
        distance = np.linalg.norm(place)
        place /= distance

        inner_product = (place[0] * direction[0]) + (place[1] * direction[1])

        target_place = pos_puck[:2] - direction * 0.07

        # max_y = pos_goal[1] - 0.15
        # if target_place[1] > max_y:
        #     target_place[1] = max_y

        # if inner_product > -0.95:
        #     return [target_place[0], target_place[1], pos_puck[2] + 0.15]

        if abs(pos_curr[2] - pos_puck[2]) > 0.04:
            return [target_place[0], target_place[1], pos_puck[2] + 0.03]

        # if pos_goal[1] > max_y:
        #     pos_goal[1] = max_y

        
        if dir_dist > 0.02:
            return pos_curr + 0.02 * np.array([direction[0], direction[1], pos_goal[2]])
        return pos_goal

    @staticmethod
    def _grab_effort(o_d):
        return 1.0
        # pos_curr = o_d['hand_pos']
        # pos_puck = o_d['puck_pos']

        # if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.02 or abs(pos_curr[2] - pos_puck[2]) > 0.10:
        #     return 0.
        # # While end effector is moving down toward the puck, begin closing the grabber
        # else:
        #     return 0.6

    def get_adaptive_action(self, obs):
        o_d = self._parse_obs(obs)

        if self.adaptive_n is None:
            self.adaptive_n = np.random.randint(1, 14)
            margin = 0.04
            rod_lo = -self.env_pm - 0.2 + margin
            rod_hi = -self.env_pm + 0.2 - margin
            offsets = []
            while len(offsets) < self.adaptive_n:
                cand = np.random.uniform(rod_lo, rod_hi)
                if abs(cand) >= 0.03:
                    offsets.append(cand)
            self.wrong_offsets = offsets

        action = Action({'delta_pos': np.arange(3), 'grab_effort': 3})
        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._adaptive_desired_pos(o_d), p=10.)
        action['grab_effort'] = 1.0
        return action.array

    def _adaptive_desired_pos(self, o_d):
        pos_curr = o_d['hand_pos']
        pos_obj  = o_d['puck_pos']
        pos_goal = o_d['goal_pos']

        self.current_is_keyframe = False
        self.keyframe_type = None

        is_success = (self.adaptive_phase >= self.adaptive_n)

        if self.external_target is not None:
            target_obj = np.asarray(self.external_target, dtype=np.float32)
        elif is_success:
            target_obj = pos_obj.copy()
        else:
            off = self.wrong_offsets[self.adaptive_phase]
            target_obj = pos_obj + np.array([-off, 0., 0.])
        self.target = target_obj

        if self.push_unit_xy is None:
            d = pos_goal[:2] - target_obj[:2]
            n = np.linalg.norm(d)
            self.push_unit_xy = d / n if n > 1e-6 else np.array([0., 1.])

        contact_xy = target_obj[:2] - 0.07 * self.push_unit_xy
        contact_z  = pos_obj[2] + 0.03

        if self.sub_phase == 0:
            if pos_curr[2] < 0.13 and np.linalg.norm(pos_curr[:2] - contact_xy) > 0.02:
                return np.array([pos_curr[0], pos_curr[1], 0.15])
            if np.linalg.norm(pos_curr[:2] - contact_xy) > 0.02:
                return np.array([contact_xy[0], contact_xy[1], 0.15])
            self.current_is_keyframe = True
            self.keyframe_type = 'hover'
            self.sub_phase = 1

        if self.sub_phase == 1:
            if abs(pos_curr[2] - contact_z) > 0.02:
                return np.array([contact_xy[0], contact_xy[1], contact_z])
            self.sub_phase = 2
            self.push_start_xy = pos_curr[:2].copy()

        if self.sub_phase == 2:
            if is_success or self.external_target is not None:
                step_xy = pos_goal[:2].copy()
            else:
                step_xy = self.push_start_xy + 0.20 * self.push_unit_xy
            step_xy = np.array([
                np.clip(step_xy[0], -0.45, 0.45),
                np.clip(step_xy[1], 0.42, 0.95),
            ])
            if np.linalg.norm(pos_curr[:2] - step_xy) > 0.02:
                return np.array([step_xy[0], step_xy[1], contact_z])

            if self.release_countdown == 0:
                self.current_is_keyframe = True
                self.keyframe_type = 'lift'
                self.release_countdown = 15
            self.release_countdown -= 1
            if self.release_countdown == 0:
                self.attempt_done = True
                self.adaptive_phase += 1
                self.sub_phase = 0
                self.push_unit_xy = None
                self.push_start_xy = None
            return np.array([step_xy[0], step_xy[1], contact_z])
