"""Scripted expert for ``brick-slide-v2`` (Slide Brick as used in the SCOUT paper).

Copied verbatim from the original (mujoco-py) Meta-World fork the released
checkpoint was trained with; only the class names differ, so that they do not
clash with :mod:`metaworld.policies.sawyer_brick_slide_v2_policy`, which belongs
to the unrelated ``slide-brick-v2`` variant.

* :class:`SawyerBrickSlideIsePolicy` holds the friction -> apex-height lookup
  table (``self.highest``).
* :class:`SawyerBrickSlideIsePolicyAdaptive` tries ``N ~ U{1..8}`` wrong apex
  heights and then the correct one (data collection).
"""
import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerBrickSlideIsePolicy(Policy):
    def __init__(self, p_friction=0, highest=None):
        self.checkpoint=False
        
        
        if highest is None:
            if p_friction >= 0.395:
                self.highest = 0.210
            elif p_friction < 0.395 and p_friction >= 0.385:
                self.highest = 0.200
            elif p_friction < 0.385 and p_friction >= 0.37:
                self.highest = 0.195
            elif p_friction < 0.37 and p_friction >= 0.355:
                self.highest = 0.190
            elif p_friction < 0.355 and p_friction >= 0.345:
                self.highest = 0.186
            elif p_friction < 0.345 and p_friction >= 0.33:
                self.highest = 0.176
            elif p_friction < 0.33 and p_friction >= 0.31:
                self.highest = 0.173
            elif p_friction < 0.31 and p_friction >= 0.29:
                self.highest = 0.167
            elif p_friction < 0.29 and p_friction >= 0.275:
                self.highest = 0.164
            elif p_friction < 0.275 and p_friction >= 0.265:
                self.highest = 0.159
            elif p_friction < 0.265 and p_friction >= 0.255:
                self.highest = 0.150
            elif p_friction < 0.255 and p_friction >= 0.245:
                self.highest = 0.146
            elif p_friction < 0.245:
                self.highest = 0.140
                
        else:
            self.highest = highest
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        # print(obs)
        # print(obs.shape)
        parsed_obs = {
            'hand_pos': obs[:3],
            'gripper': obs[3],
            'lid_pos': obs[4:7],
            'extra_info_1': obs[7:-3],
            'box_pos': obs[-3:-1],
            'extra_info_2': obs[-1],
        }
        
        # print(parsed_obs['lid_pos'])
        
        # print(parsed_obs)
        # assert False
        return parsed_obs

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=5.)
        # print("DELTA_POS", action['delta_pos'])
        # action['grab_effort'] = self._grab_effort(o_d)
        action['grab_effort'] = 0.

        return action.array

    def _desired_pos(self, o_d):
        push_r = 0.08
        # print(o_d)
        pos_curr = o_d['hand_pos']
        pos_lid = o_d['lid_pos'] + np.array([.0, .0, .07])
        push_pos = pos_lid + np.array([0, push_r, 0])
        pos_box = np.array([*o_d['box_pos'], 0.15]) + np.array([.0, .0, .0])
        # 1. move to pos_lid + [.0 .1 .0]
        # 2. move to goal

        # return np.array([0.0, 0.5, 0.1])
        # print("HEIGHT", pos_lid[2], self.checkpoint)
        checkpoint = (pos_lid[2] - 0.07) > self.highest
        # checkpoint = (pos_lid[2] - 0.07) > 0.14
        # print("CHECKPOINT", self.checkpoint, checkpoint, pos_lid[2])
        self.last_pos = pos_lid[2]
        
        if not self.checkpoint and not checkpoint:
            # print("ORIGIN")
            return o_d['lid_pos'] + np.array([0.2, 0.0, -0.1])
        elif checkpoint or self.checkpoint:
            self.checkpoint = True
            return o_d['lid_pos'] + np.array([-0.1, 0.0, 0.1])
        else:
            # print("hahaha")
            return o_d['lid_pos'] + np.array([0.0, -0.1, 0.0])

    @staticmethod
    def _grab_effort(o_d):
        return 1.

# Range over which the `highest` lookup table is defined (see
# SawyerBrickSlideIsePolicy.__init__). Used to sample wrong-attempt Z targets.
HIGHEST_RANGE = (0.140, 0.210)
WRONG_HIGHEST_RANGE = (0.110, 0.260)  # wider than valid → some failures over,
                                      # some under the valid range
MARGIN = 0.015                         # min distance from true_highest


class SawyerBrickSlideIsePolicyAdaptive(SawyerBrickSlideIsePolicy):
    """Adaptive multi-attempt variant of the brick-slide policy.

    Mirrors `SawyerPickBarV2Policy.get_adaptive_action`: tries N wrong values
    of `highest` (the friction-dependent lift threshold), then the correct
    one from the friction→highest lookup. Each attempt drives a three-stage
    motion (tilt-up → push → wait-for-still), snapshotting a `'hover'`
    keyframe at the start of an attempt and a `'lift'` keyframe once the
    brick has come to rest. The caller resets the brick on `attempt_done`.

    The exposed per-attempt "action" is a 3-D lift-apex point
    `(brick_x_init, brick_y_init, target_highest)` — Z carries the
    friction-dependent signal; X/Y anchor the action to the brick.
    """

    def __init__(self, p_friction=0, highest=None):
        super().__init__(p_friction=p_friction, highest=highest)

        # multi-attempt state
        self.adaptive_phase = 0
        self.adaptive_n = None
        self.wrong_highests = None
        self.sub_phase = 0
        self.attempt_done = False
        self.current_is_keyframe = False
        self.keyframe_type = None

        # per-attempt scratch state
        self.lift_steps = 0
        self.push_steps = 0
        self.still_count = 0
        self.last_pos = None
        self.target_highest = None
        self.target_lift_apex = None
        # plateau detection for sub_phase 0 — guards against unreachable
        # wrong_highest values where the brick caps out below target and the
        # hand would otherwise chase/carry it off the table until timeout.
        self.peak_z = -np.inf
        self.stall_steps = 0

        # captured lazily on the first observation
        self.brick_init_xy = None
        self.init_hand_pos = None

    def _init_adaptive(self, brick_xy, hand_pos):
        self.adaptive_n = int(np.random.randint(1, 9))
        lo, hi = WRONG_HIGHEST_RANGE
        wrongs = []
        while len(wrongs) < self.adaptive_n:
            cand = float(np.random.uniform(lo, hi))
            if abs(cand - float(self.highest)) >= MARGIN:
                wrongs.append(cand)
        self.wrong_highests = wrongs
        self.brick_init_xy = np.asarray(brick_xy, dtype=np.float32).copy()
        self.init_hand_pos = np.asarray(hand_pos, dtype=np.float32).copy()

    def get_gt_action(self, obs):
        """Ground-truth 3-D lift-apex action for the current friction."""
        o_d = self._parse_obs(obs)
        bx, by = o_d['lid_pos'][:2]
        return np.array([bx, by, float(self.highest)], dtype=np.float32)

    def get_adaptive_action(self, obs):
        o_d = self._parse_obs(obs)
        if self.adaptive_n is None:
            self._init_adaptive(o_d['lid_pos'][:2], o_d['hand_pos'])

        action = Action({'delta_pos': np.arange(3), 'grab_effort': 3})
        action['delta_pos'] = move(o_d['hand_pos'],
                                   to_xyz=self._adaptive_desired_pos(o_d),
                                   p=5.)
        action['grab_effort'] = 0.
        return action.array

    def _adaptive_desired_pos(self, o_d):
        pos_brick = o_d['lid_pos']
        brick_z = pos_brick[2]

        self.current_is_keyframe = False
        self.keyframe_type = None

        # pick this attempt's target_highest (true or wrong)
        is_success = (self.adaptive_phase >= self.adaptive_n)
        if is_success:
            self.target_highest = float(self.highest)
        else:
            self.target_highest = float(self.wrong_highests[self.adaptive_phase])

        # 3-D lift-apex point — anchored to the brick's init xy, Z = target
        self.target_lift_apex = np.array([
            self.brick_init_xy[0],
            self.brick_init_xy[1],
            self.target_highest,
        ], dtype=np.float32)

        # ── sub_phase 0: tilt-up ─────────────────────────────────────────
        # Hand drives toward a position below the brick edge; the floor /
        # support stops it, so the hand pushes the brick upward. Transition
        # when brick_z exceeds target_highest (or on timeout). The 'hover'
        # keyframe fires on the first step of the attempt so the dataset
        # snapshots the input PCD before any motion.
        if self.sub_phase == 0:
            if self.lift_steps == 0:
                self.current_is_keyframe = True
                self.keyframe_type = 'hover'
                self.peak_z = brick_z
                self.stall_steps = 0

            # Track peak brick_z. If it stops increasing, the wrong_highest
            # target is unreachable for this friction — exit so the hand can
            # retreat instead of carrying the brick across the table.
            if brick_z > self.peak_z + 5e-4:
                self.peak_z = brick_z
                self.stall_steps = 0
            else:
                self.stall_steps += 1

            if brick_z > self.target_highest:
                self.sub_phase = 1
                self.push_steps = 0
            elif self.lift_steps >= 30 and self.stall_steps >= 15:
                # plateau: brick_z capped below target → retreat
                self.sub_phase = 1
                self.push_steps = 0
            elif self.lift_steps > 250:
                # absolute backstop
                self.sub_phase = 1
                self.push_steps = 0
            else:
                self.lift_steps += 1
                return pos_brick + np.array([0.2, 0.0, -0.1])

        # ── sub_phase 1: retreat + wait for brick to settle ─────────────
        # No separate push phase — the hand sweep from sub_phase 0's +x side
        # toward this retreat target (in -x, high in z) sweeps across the
        # brick top and imparts the -x momentum, while moving the hand out
        # of the brick's slide path before it can come back down and bump.
        # 3D delta so 'lift' doesn't fire while the brick is still falling
        # vertically.
        if self.sub_phase == 1:
            delta = float(np.linalg.norm(pos_brick - self.last_pos)) if self.last_pos is not None else 1.0
            if delta < 1e-4:
                self.still_count += 1
            else:
                self.still_count = 0
            self.last_pos = pos_brick.copy()

            if self.still_count >= 10:
                self.current_is_keyframe = True
                self.keyframe_type = 'lift'
                self.attempt_done = True
                self.adaptive_phase += 1
                self.sub_phase = 0
                self.lift_steps = 0
                self.last_pos = None
                self.still_count = 0
                self.peak_z = -np.inf
                self.stall_steps = 0
            return np.array([self.brick_init_xy[0] - 0.4,
                             self.brick_init_xy[1],
                             0.45], dtype=np.float32)
