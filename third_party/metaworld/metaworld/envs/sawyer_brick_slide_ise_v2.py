"""Slide Brick as used in the SCOUT paper (``brick-slide-v2``).

The arm tilts the brick up a ramp to some apex height and releases it; the
brick has to slide back and come to rest inside a narrow height window.  The
right apex depends on the ramp friction, which is drawn per episode and is not
observable.

This is the task the released ``brick.pt`` checkpoint was trained on.  It was
built on the original (mujoco-py, MuJoCo 2.1) Meta-World and only reproduces
under MuJoCo <= 2.3.7: from MuJoCo 3.0 the arm settles in a different joint
configuration during the reset and the slide dynamics change, so the expert's
ground-truth apex succeeds 7% of the time instead of 90%.  The three
differences from the MuJoCo-3 Meta-World base class below are what is needed,
on top of MuJoCo 2.3.7, to reproduce the original environment (bit-exact reset
state, same ground-truth success rate):

* the mocap weld uses the identity relative pose and ``torquescale = 1``
  (Meta-World v3 uses ``torquescale = 5``);
* ``step`` builds the observation straight after ``mj_step``, without the extra
  ``mj_forward`` and the observation clipping of the v3 base class;
* the success test and reset of the original task (below).

``slide-brick-v2`` (:mod:`metaworld.envs.sawyer_brick_slide_v2`) is a different,
unused variant of the task (goal-distance success test) and is left untouched.
"""

import mujoco
import numpy as np

from metaworld.envs.sawyer_brick_slide_v2 import SawyerBrickSlideEnvV2

# Geoms whose sliding friction is the hidden parameter (the three ramp segments).
RAMP_GEOMS = (36, 37, 38)
# Brick body geom (the point-cloud object); geom 41 is the small green marker.
BRICK_GEOM = 40
FRICTION_RANGE = (0.245, 0.395)


class SawyerBrickSlideIseEnvV2(SawyerBrickSlideEnvV2):
    # Multi-attempt episodes run up to 14 attempts x 600 steps; lift the
    # per-episode step cap (the original collector/rollout used 100000).
    max_path_length = 100000

    def __init__(self, friction=0.3, **kwargs):
        super().__init__(friction=friction, **kwargs)
        self.max_path_length = 100000
        self.friction = float(friction)

    def set_friction(self, friction):
        """Change the hidden ramp friction (must be called before ``reset``)."""
        self.friction = float(friction)
        for g in RAMP_GEOMS:
            self.model.geom_friction[g][0] = self.friction

    def brick_init_pos(self):
        """Brick rest pose before physics warm-up (the XML body position)."""
        return np.array([-0.2, 0.7, 0.06], dtype=np.float64) + self.random_noise

    # ── original-Meta-World semantics ────────────────────────────────────────
    def reset_mocap_welds(self):
        if self.model.nmocap > 0 and self.model.eq_data is not None:
            for i in range(self.model.eq_data.shape[0]):
                if self.model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                    self.model.eq_data[i] = np.array(
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    def step(self, action):
        assert len(action) == 4, f"Actions should be size 4, got {len(action)}"
        if not self._set_task_called:
            raise RuntimeError("You must call env.set_task before using env.step")
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]], n_frames=self.frame_skip)
        self.curr_path_length += 1
        obs = self._get_obs()
        self._last_stable_obs = obs
        reward, info = self.evaluate_state(obs, action)
        truncate = self.curr_path_length >= self.max_path_length
        return np.array(obs, dtype=np.float64), reward, False, truncate, info

    def reset_model(self):
        self._reset_hand()
        self.threshold = False
        self.last_height = 0
        self._target_pos = self.goal.copy()
        self.obj_init_pos = np.array(self.init_config['obj_init_pos'])
        self.obj_init_angle = self.init_config['obj_init_angle']
        # Meta-World's task sampling needs a random goal per reset; the success
        # test does not use it and the brick is not moved.
        self._target_pos = self._get_state_rand_vec()[3:].copy()
        return self._get_obs()

    def compute_reward(self, actions, obs):
        # Success: after being lifted above 0.14 m the brick comes to rest
        # (height change < 3e-5 between steps) at a height in (0.073, 0.093).
        height = obs[6]
        if height > 0.14:
            self.threshold = True
        success = ((0.073 < height < 0.093) and self.threshold
                   and abs(self.last_height - height) < 0.00003)
        self.last_height = height
        reward = 10.0 if success else 0.0
        return reward, 0.0, 0.0, 0.0, success
