"""PyBullet rendering of a PartNet-Mobility object into a segmented point cloud.

Vendored and trimmed from ``rpad.partnet_mobility_utils.render.pybullet``.
"""

from __future__ import annotations

import functools
import os
import sys
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pybullet as p

PLANE_URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "plane.urdf")
RENDER_WIDTH = 640
RENDER_HEIGHT = 480
CAMERA_INTRINSICS = np.array([[450, 0, RENDER_WIDTH / 2],
                              [0, 450, RENDER_HEIGHT / 2],
                              [0, 0, 1]], dtype=float)
T_CAMGL_2_CAM = np.diag([1.0, -1.0, -1.0, 1.0])


def sample_az_ele(radius, az_lo, az_hi, ele_lo, ele_hi, seed=None):
    """Random (azimuth, elevation) on a sphere of ``radius`` -> cartesian."""
    rng = np.random.default_rng(seed)
    az, el = rng.uniform(az_lo, az_hi), rng.uniform(ele_lo, ele_hi)
    return (-radius * np.cos(el) * np.sin(az), radius * np.cos(az), radius * np.sin(el) * np.sin(az))


def get_obj_z_offset(object_id, client_id, starting_min=0.0):
    bboxes = [p.getAABB(object_id, physicsClientId=client_id)]
    for i in range(p.getNumJoints(object_id, physicsClientId=client_id)):
        bboxes.append(p.getAABB(object_id, i, physicsClientId=client_id))
    return functools.reduce(lambda a, b: min(a, b[0][2]), bboxes, starting_min)


@contextmanager
def suppress_stdout():
    """Silence PyBullet's C-level chatter while loading URDFs."""
    fd = sys.stdout.fileno()

    def redirect(to):
        sys.stdout.close()
        os.dup2(to.fileno(), fd)
        sys.stdout = os.fdopen(fd, "w")

    with os.fdopen(os.dup(fd), "w") as old:
        with open(os.devnull, "w") as devnull:
            redirect(devnull)
        try:
            yield
        finally:
            redirect(old)


def depth_to_pointcloud(depth, intrinsics):
    h, w = depth.shape
    px, py = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    return np.float32([px, py, depth]).transpose(1, 2, 0)


class Camera:
    def __init__(self, pos, render_height=RENDER_HEIGHT, render_width=RENDER_WIDTH,
                 znear=0.01, zfar=6, intrinsics=CAMERA_INTRINSICS, target=None):
        self.intrinsics = intrinsics
        self.znear, self.zfar = znear, zfar
        self.render_width, self.render_height = render_width, render_height
        fov_v = np.degrees(np.arctan((render_height / 2) / intrinsics[0][0]) * 2)
        self.proj_list = p.computeProjectionMatrixFOV(fov_v, render_width / render_height, znear, zfar)
        self.target = [0, 0, 0.5] if target is None else target
        self.set_camera_position(pos)

    def set_camera_position(self, pos):
        self.view_list = p.computeViewMatrix(pos, self.target, [0.0, 0.0, 1.0])
        self.T_camgl2world = np.asarray(self.view_list).reshape(4, 4).T
        self.T_world2camgl = np.linalg.inv(self.T_camgl2world)
        self.T_world2cam = self.T_world2camgl @ T_CAMGL_2_CAM

    def render(self, client_id, has_plane=True) -> Tuple[np.ndarray, ...]:
        """Returns ``(rgb, depth, seg, P_cam, P_world, pc_seg, segmap)``."""
        _, _, rgb, zbuf, seg = p.getCameraImage(self.render_width, self.render_height, self.view_list,
                                                self.proj_list, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                                physicsClientId=client_id)
        if isinstance(rgb, tuple):   # some platforms return flat tuples
            rgb = np.asarray(rgb).reshape(self.render_height, self.render_width, 4)
            zbuf = np.asarray(zbuf).reshape(self.render_height, self.render_width)
            seg = np.asarray(seg).reshape(self.render_height, self.render_width)

        zfar, znear = self.zfar, self.znear
        depth = (2.0 * znear * zfar) / (zfar + znear - (2.0 * zbuf - 1.0) * (zfar - znear))

        P_cam = depth_to_pointcloud(depth, self.intrinsics)
        fg = seg > 0 if has_plane else seg > -1
        pc_seg = seg[fg].flatten()
        P_cam = P_cam[fg].reshape(-1, 3)
        Ph_cam = np.concatenate([P_cam, np.ones((len(P_cam), 1))], axis=1)
        P_world = (self.T_world2cam @ Ph_cam.T).T[:, :3]

        # PyBullet packs (object id, link index + 1) into the segmentation label.
        segmap = {label: (label & ((1 << 24) - 1), (label >> 24) - 1) for label in np.unique(seg)}
        return rgb, depth, seg, P_cam, P_world, pc_seg, segmap


class PMRenderEnv:
    """A PartNet-Mobility object on a ground plane in its own PyBullet client."""

    def __init__(self, obj_id: str, dataset_path: str, camera_pos: List = (-2, 0, 2),
                 gui: bool = False, with_plane: bool = True):
        self.with_plane = with_plane
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        if with_plane:
            self.plane_id = p.loadURDF(PLANE_URDF, physicsClientId=self.client_id)
        p.setGravity(0, 0, 0, self.client_id)

        self.obj_id_str = obj_id
        with suppress_stdout():
            self.obj_id = p.loadURDF(os.path.join(dataset_path, obj_id, "mobility.urdf"),
                                     useFixedBase=True, physicsClientId=self.client_id)

        # Rest the object on the plane.
        self.minz = get_obj_z_offset(self.obj_id, self.client_id)
        p.resetBasePositionAndOrientation(self.obj_id, posObj=[0, 0, -self.minz], ornObj=[0, 0, 0, 1],
                                          physicsClientId=self.client_id)
        self.T_world_base = np.eye(4)
        self.T_world_base[2, 3] = -self.minz

        self.camera = Camera(pos=list(camera_pos), znear=0.01, zfar=10)

        self.link_name_to_index: Dict[str, int] = {
            p.getBodyInfo(self.obj_id, physicsClientId=self.client_id)[0].decode("UTF-8"): -1}
        self.jn_to_ix: Dict[str, int] = {}
        for i in range(p.getNumJoints(self.obj_id, physicsClientId=self.client_id)):
            info = p.getJointInfo(self.obj_id, i, physicsClientId=self.client_id)
            self.link_name_to_index[info[12].decode("UTF-8")] = i
            if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self.jn_to_ix[info[1].decode("UTF-8")] = i

    def render(self):
        return self.camera.render(self.client_id, self.with_plane)

    def set_camera(self, camera_xyz: Union[str, Tuple[float, float, float]], seed: Optional[int] = None):
        if camera_xyz == "random":
            camera_xyz = sample_az_ele(np.sqrt(8), np.deg2rad(30), np.deg2rad(150),
                                       np.deg2rad(30), np.deg2rad(60), seed=seed)
        self.camera.set_camera_position(list(camera_xyz))

    def get_joint_angles(self) -> Dict[str, float]:
        angles = {}
        for i in range(p.getNumJoints(self.obj_id, self.client_id)):
            info = p.getJointInfo(self.obj_id, i, self.client_id)
            angles[info[1].decode("UTF-8")] = p.getJointState(self.obj_id, i, self.client_id)[0]
        return angles

    def get_joint_ranges(self) -> Dict[str, Tuple[float, float]]:
        ranges = {}
        for i in range(p.getNumJoints(self.obj_id, self.client_id)):
            info = p.getJointInfo(self.obj_id, i, self.client_id)
            ranges[info[1].decode("UTF-8")] = (info[8], info[9])
        return ranges

    def close(self):
        p.disconnect(self.client_id)
