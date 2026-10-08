"""Analytic articulation of a point cloud and ground-truth flow.

Vendored from ``rpad.partnet_mobility_utils.articulate`` and
``flowbot3d.datasets.flow_dataset.compute_normalized_flow``; the trimesh
dependency is replaced by scipy.
"""

from __future__ import annotations

import copy
from functools import reduce
from typing import Dict, List, Sequence, Union

import numpy as np
from scipy.spatial.transform import Rotation

from scout.envs.door.partnet_mobility import Joint, PMObject


def fk(chain: List[Joint], joint_angles: Sequence[float]) -> np.ndarray:
    """Forward kinematics along ``chain``: T_base_endlink."""

    def joint_transform(joint: Joint, angle: float) -> np.ndarray:
        T_link_child = np.eye(4)
        if joint.origin is not None:
            xyz, rpy = joint.origin
            T_link_child[:3, 3] = xyz
            T_link_child[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()

        T_art = np.eye(4)
        axis = joint.axis if joint.axis is not None else np.asarray([1.0, 0.0, 0.0])
        if joint.type in ("revolute", "continuous"):
            if angle != 0.0:
                T_art[:3, :3] = Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix()
        elif joint.type == "prismatic":
            T_art[:3, 3] = axis / np.linalg.norm(axis) * angle
        return T_link_child @ T_art

    tforms = [joint_transform(j, a) for j, a in zip(chain, joint_angles)]
    return reduce(lambda a, b: a @ b, tforms, np.eye(4))


def articulate_points(P_world: np.ndarray, T_world_base: np.ndarray, chain: List[Joint],
                      current_ja: Sequence[float], target_ja: Sequence[float]) -> np.ndarray:
    """Move points attached to the end link of ``chain`` from ``current_ja`` to ``target_ja``."""
    if P_world.shape == (3,):
        P_world = P_world[None]
    assert P_world.ndim == 2 and P_world.shape[1] == 3
    assert len(chain) == len(current_ja) == len(target_ja)
    N = len(P_world)
    Ph = np.concatenate([P_world, np.ones((N, 1))], axis=1)

    T_base_end = fk(chain, current_ja)
    T_world_end = T_world_base @ T_base_end
    Ph_end = (np.linalg.inv(T_world_end) @ Ph.T).T                 # points in end-link frame

    T_end_endnew = np.linalg.inv(T_base_end) @ fk(chain, target_ja)
    Ph_world_new = (T_world_end @ (T_end_endnew @ Ph_end.T)).T
    return Ph_world_new[:, :3]


def articulate_joint(obj: PMObject, current_jas: Dict[str, float], link_to_actuate: str,
                     amount: float, pos: np.ndarray, seg: np.ndarray, labelmap: Dict[str, int],
                     T_world_base: np.ndarray) -> np.ndarray:
    """Articulate ``link_to_actuate`` (and its descendants) by ``amount``; other points stay put."""
    chain = obj.obj.get_chain(link_to_actuate)
    c_jas = [current_jas[j.name] for j in chain]
    target = copy.deepcopy(current_jas)
    target[obj.obj.get_joint_by_child(link_to_actuate).name] += amount
    t_jas = [target[j.name] for j in chain]

    links = [link_to_actuate] + obj.obj.descendants[link_to_actuate]
    seg_ids = np.asarray([labelmap[l] for l in links])
    to_act = (seg.reshape(-1, 1) == seg_ids.reshape(1, -1)).any(axis=-1)

    pos_new = np.copy(pos)
    if to_act.any():
        pos_new[to_act] = articulate_points(pos[to_act], T_world_base, chain, c_jas, t_jas)
    return pos_new


def compute_normalized_flow(P_world: np.ndarray, T_world_base: np.ndarray, current_jas: Dict[str, float],
                            pc_seg: np.ndarray, labelmap: Dict[str, int], obj: PMObject,
                            linknames: Union[str, Sequence[str]] = "all") -> np.ndarray:
    """Ground-truth articulation flow: each movable link opened by 0.01, summed and
    normalised so the largest per-point magnitude is 1."""
    if linknames == "all":
        linknames = obj.movable_links()
    flow = np.zeros_like(P_world)
    for name in linknames:
        flow += articulate_joint(obj, current_jas, name, 0.01, P_world, pc_seg, labelmap, T_world_base) - P_world
    return flow / (np.linalg.norm(flow, axis=-1).max() + 1e-6)
