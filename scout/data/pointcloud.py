"""Point-cloud helpers shared by datasets and rollouts."""

import numpy as np


def sample_pcd(pts, n_pts: int = 1200, rng=None) -> np.ndarray:
    """Random sub-sample (or over-sample with replacement) to exactly ``n_pts``."""
    pts = np.asarray(pts, dtype=np.float32)
    n = len(pts)
    if n == 0:
        return np.zeros((n_pts, 3), dtype=np.float32)
    rng = np.random if rng is None else rng
    idx = rng.choice(n, n_pts, replace=(n < n_pts))
    return pts[idx]


def normalize_pcd(pcd: np.ndarray):
    """Centre on the mean and scale by the max radius.

    Returns ``(pcd_norm, center, scale)``.  The same ``(center, scale)`` must
    be applied to every point cloud and action of a sample so that history
    and query share one frame.
    """
    center = pcd.mean(axis=0)
    scale = max(float(np.linalg.norm(pcd - center, axis=1).max()), 1e-6)
    return (pcd - center) / scale, center, scale


def apply_frame(x, center, scale):
    return (np.asarray(x, dtype=np.float32) - center) / scale


def undo_frame(x, center, scale):
    return np.asarray(x, dtype=np.float32) * scale + center
