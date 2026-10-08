#!/usr/bin/env python
"""Check that headless MuJoCo rendering works for the Meta-World (ISE) tasks.

    python scripts/diag_render.py                      # default: pick-bar-v2
    python scripts/diag_render.py --env faucet-open-v3

The point clouds are built from a segmentation render, which needs a working
OpenGL backend.  On some headless NVIDIA datacenter GPUs (seen on H200, driver
570) the NVIDIA EGL driver returns corrupted images; the segmentation render then
fails (an ``IndexError`` inside ``mujoco.Renderer.render()``, which SCOUT re-raises
as a ``RuntimeError`` with this hint).  Mesa's software EGL is a
working fallback:

    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
"""

import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402


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
    renderer.close()
    env.close()

    if ok:
        print("PASS: rendering works")
    else:
        print("FAIL: try Mesa's software EGL:\n"
              "  export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json\n"
              "(or MUJOCO_GL=osmesa if libOSMesa is installed) and re-run this script.")
        sys.exit(1)


if __name__ == "__main__":
    main()
