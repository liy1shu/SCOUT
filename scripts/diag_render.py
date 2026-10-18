#!/usr/bin/env python
"""Check that headless MuJoCo rendering works for the Meta-World (ISE) tasks.

    python scripts/diag_render.py                      # default: pick-bar-v2
    python scripts/diag_render.py --env faucet-open-v3
    python scripts/diag_render.py --env brick-slide-v2 # in the scout-brick env (MuJoCo 2.3.7)

The point clouds are built from a segmentation render, which needs a working
OpenGL backend.  On some headless NVIDIA datacenter GPUs (seen on H200, driver
570) the NVIDIA EGL driver returns corrupted images; the segmentation render then
fails (an ``IndexError`` inside ``mujoco.Renderer.render()``, which SCOUT re-raises
as a ``RuntimeError`` with this hint).  Mesa's software EGL is a
working fallback:

    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json

The brick task runs on MuJoCo 2.3.7, whose segmentation rendering does not work
on NVIDIA EGL; its point clouds are built from depth renders instead, so for
``--env brick-slide-v2`` this script checks that brick cloud.
"""

import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scout.envs.metaworld.pointcloud import close_renderer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default="pick-bar-v2")
    args = parser.parse_args()

    import gymnasium as gym
    import metaworld  # noqa: F401  (registers the Meta-World envs)
    import mujoco

    print(f"python {sys.version.split()[0]} | mujoco {mujoco.__version__} | gymnasium {gym.__version__} | "
          f"MUJOCO_GL={os.environ.get('MUJOCO_GL')} | "
          f"__EGL_VENDOR_LIBRARY_FILENAMES={os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES')}")

    if args.env == "brick-slide-v2":
        return check_brick()

    env = gym.make("Meta-World/MT1", env_name=args.env)
    env.reset()
    base = env.unwrapped
    renderer = mujoco.Renderer(base.model, 480, 480)
    renderer.update_scene(base.data, camera="corner")
    rgb = renderer.render()

    from OpenGL.GL import GL_RENDERER, GL_VENDOR, glGetString
    print(f"OpenGL: {glGetString(GL_VENDOR).decode()} | {glGetString(GL_RENDERER).decode()}")

    ok = True
    renderer.enable_segmentation_rendering()
    renderer.update_scene(base.data, camera="corner")
    try:
        seg = renderer.render()
        ids = np.unique(seg[..., 0])
        print(f"segmentation: {len(ids)} distinct geom ids (ngeom={base.model.ngeom}), rgb mean {rgb.mean():.1f}")
    except IndexError as e:
        ok = False
        print(f"segmentation FAILED ({e}): the GL backend ignored the segmentation flags.")
    close_renderer(renderer)
    env.close()

    if ok:
        print("PASS: rendering works")
    else:
        print("FAIL: try Mesa's software EGL:\n"
              "  export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json\n"
              "(or MUJOCO_GL=osmesa if libOSMesa is installed) and re-run this script.")
        sys.exit(1)


def check_brick():
    from OpenGL.GL import GL_RENDERER, GL_VENDOR, glGetString
    from scout.envs.metaworld.tasks import get_task

    task = get_task("brick")
    env, base = task.make_env("brick-slide-v2", for_rollout=True)   # raises under MuJoCo >= 3
    env.reset()
    pts = task.object_cloud(env, None)[0]
    print(f"OpenGL: {glGetString(GL_VENDOR).decode()} | {glGetString(GL_RENDERER).decode()}")
    env.close()
    # The brick at rest covers ~4000 pixels of the 640x480 corner view.
    size = pts.max(axis=0) - pts.min(axis=0) if len(pts) else np.zeros(3)
    print(f"brick cloud: {len(pts)} points, extent {np.round(size, 3)} m")
    if len(pts) > 1000 and np.all(size < 0.3):
        print("PASS: rendering works")
    else:
        print("FAIL: the brick depth render returned no (or a wrong) brick cloud.")
        sys.exit(1)


if __name__ == "__main__":
    main()
