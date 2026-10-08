"""Suction-gripper manipulation of articulated objects in PyBullet.

Vendored and trimmed from HAVE (``have.env.articulated.suction``).  The
interaction primitive is: teleport the gripper to a stand-off pose in front
of a contact point, approach along the contact normal until touching,
activate suction, then pull along a direction for a fixed number of steps.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

from scout.envs.door.articulate import compute_normalized_flow
from scout.envs.door.partnet_mobility import PMObject
from scout.envs.door.render import PMRenderEnv
from scout.envs.door.suction_gripper import FloatingSuctionGripper

GRIPPER_HOME = ([-1, 0.6, 0.8], p.getQuaternionFromEuler([0, np.pi / 2, 0]))
# Gripper URDF frame -> approach frame (tip points along -y of the approach frame).
_R_GRIPPER = np.array([[1, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])


def approach_pose(point: np.ndarray, contact_vector: np.ndarray, standoff: float):
    """Stand-off position and orientation for approaching ``point`` along ``-contact_vector``."""
    cv = np.asarray(contact_vector, dtype=np.float64)
    cv = cv / (np.linalg.norm(cv) + 1e-12)
    pos = np.asarray(point, dtype=np.float64) + cv * standoff
    e_y = -cv
    e_x = np.cross(-cv, np.array([0.0, 0.0, 1.0]))
    e_x = e_x / (np.linalg.norm(e_x) + 1e-12)
    e_z = np.cross(e_x, e_y)
    e_z = e_z / (np.linalg.norm(e_z) + 1e-12)
    R_approach = np.stack([e_x, e_y, e_z], axis=1)
    return pos, R.from_matrix(R_approach @ _R_GRIPPER).as_quat(), cv


class PMSuctionSim:
    def __init__(self, obj_id: str, dataset_path: str, gui: bool = False, camera_pos: List = (-2, 0, 2),
                 camera_pos_random: bool = False, gripper: bool = True):
        self.render_env = PMRenderEnv(obj_id=obj_id, dataset_path=dataset_path, gui=gui, camera_pos=camera_pos)
        if camera_pos_random:
            self.render_env.set_camera("random")
        self.gui = gui
        if gui:
            p.resetDebugVisualizerCamera(cameraTargetPosition=[0, 0, 0], cameraDistance=5, cameraYaw=220,
                                         cameraPitch=-30, physicsClientId=self.render_env.client_id)
        self.gripper = FloatingSuctionGripper(self.render_env.client_id) if gripper else None
        if self.gripper is not None:
            self.gripper.set_pose(*GRIPPER_HOME)
        # Called after every simulation step of an approach or pull (video recording).
        self.step_callback = None

    def _step(self):
        p.stepSimulation(self.client_id)
        if self.step_callback is not None:
            self.step_callback()
        if self.gui:
            time.sleep(1 / 240.0)

    @property
    def client_id(self):
        return self.render_env.client_id

    def reset(self):
        pass

    def close(self):
        p.disconnect(self.client_id)

    # ── collision filtering ──────────────────────────────────────────────────
    def _filter_pair(self, body_a, body_b, link_a, link_b):
        p.setCollisionFilterPair(bodyUniqueIdA=body_a, bodyUniqueIdB=body_b, linkIndexA=link_a,
                                 linkIndexB=link_b, enableCollision=0, physicsClientId=self.client_id)

    def disable_self_collision(self):
        """Gripper never collides with the floor."""
        for part in (self.gripper.base_id, self.gripper.body_id):
            for link in range(-1, p.getNumJoints(part, physicsClientId=self.client_id)):
                self._filter_pair(part, self.render_env.plane_id, link, -1)

    def disable_collision(self, link_id, base=True, body=True, floor=False):
        """Disable gripper/object collisions for one object link.

        ``body=False`` keeps the suction tip collidable (so it can touch the
        target link) and only filters the head's base link.  ``floor=True``
        additionally lets the link pass through the ground plane.
        """
        obj = self.render_env.obj_id
        if base:
            for link in range(-1, p.getNumJoints(self.gripper.base_id, physicsClientId=self.client_id)):
                self._filter_pair(self.gripper.base_id, obj, link, link_id)
        if body:
            for link in range(-1, p.getNumJoints(self.gripper.body_id, physicsClientId=self.client_id)):
                self._filter_pair(self.gripper.body_id, obj, link, link_id)
        else:
            self._filter_pair(self.gripper.body_id, obj, -1, link_id)
        if floor:
            self._filter_pair(self.render_env.plane_id, obj, -1, link_id)
        p.stepSimulation(physicsClientId=self.client_id)

    # ── state ────────────────────────────────────────────────────────────────
    def set_joint_state(self, link_name: str, value: float):
        p.resetJointState(self.render_env.obj_id, self.render_env.link_name_to_index[link_name],
                          value, 0.0, self.client_id)

    def get_joint_value(self, link_name: str) -> float:
        return p.getJointState(self.render_env.obj_id, self.render_env.link_name_to_index[link_name],
                               self.client_id)[0]

    def get_joint_limits(self, link_name: str) -> Tuple[float, float]:
        info = p.getJointInfo(self.render_env.obj_id, self.render_env.link_name_to_index[link_name], self.client_id)
        return info[8], info[9]

    def reset_gripper(self, target_link: str):
        """Release and park the gripper, keeping the target joint where it is."""
        curr = self.get_joint_value(target_link)
        self.gripper.release()
        self.gripper.set_pose(*GRIPPER_HOME)
        self.gripper.set_velocity([0, 0, 0], [0, 0, 0])
        self.set_joint_state(target_link, curr)

    # ── observation ──────────────────────────────────────────────────────────
    def render(self, filter_nonobj_pts: bool = False, n_pts: Optional[int] = None):
        """``(rgb, depth, seg, P_cam, P_world, pc_seg, segmap)``.

        With ``filter_nonobj_pts`` only object points are kept and ``pc_seg``
        becomes the object link index of each point.
        """
        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = self.render_env.render()
        if filter_nonobj_pts:
            pc_seg_obj = np.ones_like(pc_seg) * -1
            for k, (body, link) in segmap.items():
                if body == self.render_env.obj_id:
                    pc_seg_obj[pc_seg == k] = link
            is_obj = pc_seg_obj != -1
            P_cam, P_world, pc_seg = P_cam[is_obj], P_world[is_obj], pc_seg_obj[is_obj]
        if n_pts is not None:
            perm = np.random.permutation(len(P_world))[:n_pts]
            P_cam, P_world, pc_seg = P_cam[perm], P_world[perm], pc_seg[perm]
        return rgb, depth, seg, P_cam, P_world, pc_seg, segmap

    # ── interaction ──────────────────────────────────────────────────────────
    def teleport(self, points, contact_vectors, standoff_d: float = 0.2, target_link: Optional[str] = None,
                 max_steps: int = 500, print_log: bool = False, **_) -> Tuple[int, bool]:
        """Try candidate (point, direction) pairs in order until the tip makes contact.

        Returns ``(index of the successful candidate, True)`` or ``(-1, False)``.
        Failed approaches restore the target joint angle.
        """
        orig_angle = self.get_joint_value(target_link) if target_link is not None else None
        for i, (point, cv) in enumerate(zip(np.asarray(points), np.asarray(contact_vectors))):
            pos, ori, cv = approach_pose(point, cv, standoff_d)
            self.gripper.set_pose(pos, ori)
            p.stepSimulation(self.client_id)

            contact = self.gripper.detect_contact(self.render_env.obj_id)
            steps = 0
            while not contact and steps < max_steps:
                self.gripper.set_velocity(-cv * 0.4, [0, 0, 0])
                self._step()
                steps += 1
                contact = self.gripper.detect_contact(self.render_env.obj_id)
            if contact:
                if print_log:
                    print("contact detected")
                return i, True
            if target_link is not None and self.get_joint_value(target_link) != orig_angle:
                self.set_joint_state(target_link, orig_angle)
        return -1, False

    def attach(self):
        self.gripper.activate(self.render_env.obj_id)

    def pull(self, direction, n_steps: int = 100) -> bool:
        d = np.asarray(direction, dtype=np.float64)
        d = d / (np.linalg.norm(d) + 1e-12)
        for _ in range(n_steps):
            self.gripper.set_velocity(d * 0.4, [0, 0, 0])
            self._step()
        return False

    def pull_with_constraint(self, direction, n_steps: int = 100, target_link: str = "",
                             constraint: bool = True) -> bool:
        """Pull; with ``constraint`` the target joint is reset to its lower limit if the
        pull pushed it outside its range.  Returns whether such a reset happened."""
        if not constraint:
            return self.pull(direction, n_steps)
        lower, upper = self.get_joint_limits(target_link)
        self.pull(direction, n_steps)
        curr = self.get_joint_value(target_link)
        if curr < lower < upper or curr > lower > upper:
            p.resetJointState(self.render_env.obj_id, self.render_env.link_name_to_index[target_link],
                              lower, 0, self.client_id)
            return True
        return False


class GTFlowModel:
    """Ground-truth articulation flow of every movable link (normalised to max 1)."""

    def __init__(self, raw_data: PMObject, env: PMSuctionSim):
        self.env = env
        self.raw_data = raw_data

    def __call__(self, obs) -> np.ndarray:
        _, _, _, _, P_world, pc_seg, _ = obs
        current_jas = {}
        for link in self.raw_data.movable_links():
            for joint in self.raw_data.obj.get_chain(link):
                current_jas[joint.name] = 0
        return compute_normalized_flow(P_world, self.env.render_env.T_world_base, current_jas, pc_seg,
                                       self.env.render_env.link_name_to_index, self.raw_data, "all")


def choose_grasp_points(pred_flow: np.ndarray, point_cloud: np.ndarray, filter_edge: bool = False, k: int = 40):
    """Top-k points by flow magnitude -> ``(indices, flows, points)``."""
    flow = np.array(pred_flow, dtype=np.float32, copy=True)
    if filter_edge:
        dists = np.linalg.norm(point_cloud[:, None, :] - point_cloud[None, :, :], axis=2)
        neighbours = np.sum(dists < np.percentile(dists, 10), axis=0)
        flow[neighbours < np.percentile(neighbours, 30)] = 0
    k = min(k, len(flow))
    idx = np.argsort(-np.linalg.norm(flow, axis=-1))[:k]
    return idx, flow[idx], point_cloud[idx]
