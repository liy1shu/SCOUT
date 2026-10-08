from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerFaucetCloseV3Policy(Policy):

    def __init__(self):
        self.checkpoint = False
        self.prev_faucet_pos = None
        self.stuck_steps = 0
        self.stuck_threshold = 20
        self.attempt_done = False

        # Adaptive state machine: N random wrong attempts then one deterministic success
        self.adaptive_phase   = 0     # attempt index; >= adaptive_n → success attempt
        self.adaptive_n       = None  # drawn lazily on first call
        self.wrong_targets    = None  # pre-sampled wrong push_goals (world coords)
        self.right_is_open    = False # close = push left; wrong side = push right
        self.sub_phase        = 0
        self.exec_steps       = 0
        self.release_countdown = 0
        self.current_is_keyframe = False
        self.keyframe_type       = None
        self.target              = np.zeros(3)
        self.external_target     = None
        self.push_right          = True

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "unused_gripper": obs[3],
            "faucet_pos": obs[4:7],
            "faucet_rot": obs[7:11],
            "unused_info": obs[11:],
        }

    def get_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)

        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        action["delta_pos"] = move(
            o_d["hand_pos"], to_xyz=self._desired_pos(o_d), p=25.0
        )
        action["grab_effort"] = 1.0

        return action.array

    @staticmethod
    def _desired_pos(o_d: dict[str, npt.NDArray[np.float64]]) -> npt.NDArray[Any]:
        pos_curr = o_d["hand_pos"]
        pos_faucet = o_d["faucet_pos"] + np.array([+0.04, 0.0, 0.03])

        if np.linalg.norm(pos_curr[:2] - pos_faucet[:2]) > 0.04:
            return pos_faucet + np.array([0.0, 0.0, 0.1])
        elif abs(pos_curr[2] - pos_faucet[2]) > 0.04:
            return pos_faucet
        else:
            return pos_faucet + np.array([-0.1, 0.05, 0.0])
    
        
        
    def get_adaptive_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)

        # Lazy init: draw N wrong attempts. Wrong push_goal is the deterministic
        # mirror of the success push_goal (success = handle + [-0.06, +0.05, +0.03],
        # wrong = handle + [+0.06, +0.05, +0.03]) — only the sign of dx flips.
        if self.adaptive_n is None:
            # Binary task: one failure already identifies the wrong side, so
            # additional wrong attempts are redundant. Heavy-bias N=1 to cut
            # collection time, but keep a small tail at 2..5 so the history
            # encoder generalizes to longer rollouts. N=0 is dropped here
            # because the training filter (ttt_with_dyn_faucet.py:50,
            # num_history > 0) discards it anyway.
            if np.random.rand() < 0.75:
                self.adaptive_n = 1
            else:
                self.adaptive_n = int(np.random.randint(2, 6))
            handle_tip_init = o_d["faucet_pos"].copy()
            wrong_target = np.array([
                handle_tip_init[0] + 0.06,
                handle_tip_init[1] + 0.05,
                handle_tip_init[2] + 0.03,
            ], dtype=np.float32)
            self.wrong_targets = [wrong_target.copy() for _ in range(self.adaptive_n)]

        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})
        action["delta_pos"] = move(
            o_d["hand_pos"], to_xyz=self._adaptive_desired_pos(o_d), p=25.0
        )
        action["grab_effort"] = 1.0
        return action.array

    def _adaptive_desired_pos(self, o_d: dict[str, npt.NDArray[np.float64]]) -> npt.NDArray[Any]:
        pos_curr   = o_d["hand_pos"]
        handle_tip = o_d["faucet_pos"]  # handle tip site; moves as the handle rotates

        self.current_is_keyframe = False
        self.keyframe_type = None

        is_success = (self.adaptive_phase >= self.adaptive_n)

        if self.external_target is not None:
            push_goal = np.asarray(self.external_target, dtype=np.float32)
        elif is_success:
            # close env: success = push left, approach from right
            contact_def = handle_tip + np.array([+0.04, 0.0, 0.03])
            push_goal   = contact_def + np.array([-0.1, 0.05, 0.0])
        else:
            push_goal = self.wrong_targets[self.adaptive_phase]

        if push_goal[0] > handle_tip[0]:  # push right → approach from left
            contact = handle_tip + np.array([-0.04, 0.0, 0.03])
        else:                              # push left → approach from right
            contact = handle_tip + np.array([+0.04, 0.0, 0.03])

        if self.sub_phase == 0:
            self.target = push_goal
            if np.linalg.norm(pos_curr[:2] - contact[:2]) > 0.04:
                return contact + np.array([0.0, 0.0, 0.15])
            else:
                self.current_is_keyframe = True
                self.keyframe_type = 'hover'
                self.sub_phase = 1

        if self.sub_phase == 1:
            if abs(pos_curr[2] - contact[2]) > 0.04:
                return contact
            else:
                self.push_right = bool(push_goal[0] > handle_tip[0])
                self.prev_faucet_pos = None
                self.stuck_steps = 0
                self.sub_phase = 2
                self.exec_steps = 0

        if self.sub_phase == 2:
            if is_success or self.external_target is not None:
                # tracking controller: chase handle tip each step during success push
                if self.push_right:
                    c = handle_tip + np.array([-0.04, 0.0, 0.03])
                    self.target = c + np.array([+0.1, 0.05, 0.0])
                else:
                    c = handle_tip + np.array([+0.04, 0.0, 0.03])
                    self.target = c + np.array([-0.1, 0.05, 0.0])
            else:
                # static wrong push_goal (handle blocked by box_left wall, won't move)
                self.target = self.wrong_targets[self.adaptive_phase]

            if self.prev_faucet_pos is not None:
                faucet_movement = np.linalg.norm(handle_tip - self.prev_faucet_pos)
                if faucet_movement < 1e-4:
                    self.stuck_steps += 1
                else:
                    self.stuck_steps = 0
            self.prev_faucet_pos = handle_tip.copy()
            self.exec_steps += 1

            done = (self.exec_steps >= 500) or (self.stuck_steps >= self.stuck_threshold)
            if not done:
                return self.target

            if self.release_countdown == 0:
                self.current_is_keyframe = True
                self.keyframe_type = 'lift'
                self.release_countdown = 15
            self.release_countdown -= 1
            if self.release_countdown == 0:
                self.attempt_done = True
                self.adaptive_phase += 1
                self.sub_phase = 0
            return self.target
