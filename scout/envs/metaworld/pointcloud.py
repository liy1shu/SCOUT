"""Segmented point clouds from a Meta-World MuJoCo scene."""

from typing import Iterable, Tuple

import mujoco
import numpy as np
import open3d as o3d

BACKGROUND_KEYWORDS = ("floor", "table", "wall", "world")
FRAME_W = 480
FRAME_H = 480


def get_point_cloud(env, renderer, object_bodies: Iterable[str], grey_filter: bool = False,
                    camera_name: str = "corner", width: int = FRAME_W, height: int = FRAME_H
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Render RGB-D + segmentation and back-project into world coordinates.

    Returns ``(robot_points, robot_colors, object_points, object_colors)``.
    Geoms whose body name is in ``object_bodies`` form the object cloud; all
    other non-background geoms form the robot cloud.  ``grey_filter`` keeps
    only near-grey object points (drops the coloured markers on the bars).
    """
    unwrapped = env.unwrapped
    model, data = unwrapped.model, unwrapped.data
    object_bodies = set(object_bodies)

    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    depth = renderer.render()
    renderer.disable_depth_rendering()

    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera_name)
    try:
        seg = renderer.render()
    except IndexError as e:
        # mujoco decodes geom ids from the rendered colours; out-of-range ids
        # mean the GL backend ignored the segmentation flags (seen with the
        # NVIDIA EGL driver on some headless datacenter GPUs, e.g. H100/H200).
        raise RuntimeError(
            "MuJoCo segmentation rendering returned invalid ids: the OpenGL backend is not "
            "rendering correctly. Run `python scripts/diag_render.py`; on NVIDIA EGL try Mesa's "
            "software EGL: export __EGL_VENDOR_LIBRARY_FILENAMES="
            "/usr/share/glvnd/egl_vendor.d/50_mesa.json (see README, Troubleshooting).") from e
    renderer.disable_segmentation_rendering()

    geom_ids = seg[:, :, 0]
    robot_mask = np.zeros((height, width), dtype=bool)
    object_mask = np.zeros((height, width), dtype=bool)
    for gid in np.unique(geom_ids):
        if gid < 0 or gid >= model.ngeom:
            continue
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]) or ""
        if any(kw in body_name.lower() for kw in BACKGROUND_KEYWORDS):
            continue
        if body_name in object_bodies:
            object_mask |= geom_ids == gid
        else:
            robot_mask |= geom_ids == gid

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    fovy = model.cam_fovy[cam_id]
    f = 0.5 * height / np.tan(fovy * np.pi / 360)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, f, f, width / 2, height / 2)

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = data.cam_xmat[cam_id].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    extrinsic[:3, 3] = data.cam_xpos[cam_id]

    def mask_to_pcd(mask, grey=False):
        d = depth.copy()
        d[~mask] = 0
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)),
            o3d.geometry.Image(np.ascontiguousarray(d).astype(np.float32)),
            depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        pcd.transform(extrinsic)
        pts = np.asarray(pcd.points, dtype=np.float32)
        clr = np.asarray(pcd.colors, dtype=np.float32)
        if grey and len(clr):
            keep = (clr.max(axis=1) - clr.min(axis=1)) < 0.15
            pts, clr = pts[keep], clr[keep]
        return pts, clr

    robot_points, robot_colors = mask_to_pcd(robot_mask)
    object_points, object_colors = mask_to_pcd(object_mask, grey=grey_filter)
    return robot_points, robot_colors, object_points, object_colors


def camera_params(base_env, camera_name="corner", width=FRAME_W, height=FRAME_H) -> dict:
    """Pinhole parameters used to project world points onto rendered frames."""
    model, data = base_env.model, base_env.data
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    f = 0.5 * height / np.tan(model.cam_fovy[cam_id] * np.pi / 360.0)
    return dict(f=f, cam_pos=data.cam_xpos[cam_id].copy(),
                cam_mat=data.cam_xmat[cam_id].reshape(3, 3).copy(),
                flip=np.diag([1.0, -1.0, -1.0]), W=width, H=height)


def project(p_world, cp):
    R_w2c = (cp["cam_mat"] @ cp["flip"]).T
    p_cam = R_w2c @ (np.asarray(p_world) - cp["cam_pos"])
    if p_cam[2] <= 0.01:
        return None
    u = int(cp["f"] * p_cam[0] / p_cam[2] + cp["W"] / 2)
    v = int(cp["f"] * p_cam[1] / p_cam[2] + cp["H"] / 2)
    if 0 <= u < cp["W"] and 0 <= v < cp["H"]:
        return (u, v)
    return None
