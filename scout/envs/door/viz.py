"""Video recording for door rollouts.

Frames are rendered from the same camera that produces the point cloud, so
the overlaid contact point / pull direction line up with what the policy saw.
The recorder hooks into ``PMSuctionSim.step_callback`` and grabs a frame
every ``record_every`` simulation steps during approach and pull.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pybullet as p

try:
    import cv2
except ImportError:   # overlays are optional
    cv2 = None


class DoorRecorder:
    def __init__(self, env, width: int = 640, height: int = 480, record_every: int = 8):
        self.env = env
        self.width, self.height = width, height
        self.record_every = record_every
        self.frames: List[np.ndarray] = []
        self.text: List[str] = []
        self.point: Optional[np.ndarray] = None
        self.direction: Optional[np.ndarray] = None
        self.history: List[np.ndarray] = []
        self._tick = 0
        cam = env.render_env.camera
        self._view = np.asarray(cam.view_list).reshape(4, 4).T
        self._proj = np.asarray(cam.proj_list).reshape(4, 4).T
        env.step_callback = self._on_sim_step

    # ── overlay state ────────────────────────────────────────────────────────
    def set_overlay(self, text: List[str], point=None, direction=None, history=None):
        self.text = list(text)
        self.point = None if point is None else np.asarray(point, dtype=float)
        self.direction = None if direction is None else np.asarray(direction, dtype=float)
        if history is not None:
            self.history = [np.asarray(h, dtype=float) for h in history]

    def _project(self, pt):
        clip = self._proj @ self._view @ np.array([pt[0], pt[1], pt[2], 1.0])
        if clip[3] <= 1e-6:
            return None
        ndc = clip[:3] / clip[3]
        u, v = int((ndc[0] + 1) / 2 * self.width), int((1 - ndc[1]) / 2 * self.height)
        if 0 <= u < self.width and 0 <= v < self.height:
            return u, v
        return None

    # ── capture ──────────────────────────────────────────────────────────────
    def _on_sim_step(self):
        self._tick += 1
        if self._tick % self.record_every == 0:
            self.capture()

    def capture(self):
        cam = self.env.render_env.camera
        _, _, rgb, _, _ = p.getCameraImage(self.width, self.height, cam.view_list, cam.proj_list,
                                           physicsClientId=self.env.client_id)
        frame = np.asarray(rgb, dtype=np.uint8).reshape(self.height, self.width, 4)[:, :, :3].copy()
        if cv2 is not None:
            frame = self._draw(frame)
        self.frames.append(frame)

    def _draw(self, frame):
        for h in self.history:
            px = self._project(h[:3])
            if px is not None:
                cv2.drawMarker(frame, px, (200, 60, 60), cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
        if self.point is not None:
            px = self._project(self.point)
            if px is not None:
                cv2.circle(frame, px, 7, (0, 255, 255), -1)
                if self.direction is not None:
                    tip = self._project(self.point + 0.4 * self.direction / (np.linalg.norm(self.direction) + 1e-8))
                    if tip is not None:
                        cv2.arrowedLine(frame, px, tip, (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.25)
        for i, txt in enumerate(self.text):
            y = 8 + (i + 1) * 20
            cv2.putText(frame, txt, (9, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    # ── output ───────────────────────────────────────────────────────────────
    def save(self, path: str, fps: int = 20):
        self.env.step_callback = None
        if not self.frames:
            return
        try:
            import imageio
        except ImportError:
            print(f"  [video] imageio not found, skipping {path}")
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=1)
        for f in self.frames:
            writer.append_data(f)
        writer.close()
        print(f"  Saved video -> {path} ({len(self.frames)} frames)")
