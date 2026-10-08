"""Simulator-agnostic inference helpers used by every rollout.

Point clouds and targets live in world coordinates on the simulator side;
these helpers move them into the frame the model was trained in (per-sample
normalisation for the ISE tasks, raw coordinates for the doors) and back.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch

from scout.data.pointcloud import apply_frame, normalize_pcd, undo_frame
from scout.models.scout import SCOUT


def _frame(eval_pcd: np.ndarray, normalize: bool):
    if normalize:
        _, center, scale = normalize_pcd(eval_pcd)
        return center, scale
    return np.zeros(3, dtype=np.float32), 1.0


def predict_target(model: SCOUT, q: torch.Tensor, eval_pcd: np.ndarray, use_argmax: bool,
                   normalize: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the policy on one observation.

    Returns ``(target [A], weights [N*K], means [N*K, A] in world coordinates,
    point_weights [N])`` where point_weights sums the K components of each
    point (for visualisation).  ``target`` is the highest-weight proposal or
    a proposal sampled from the mixture weights.
    """
    N = eval_pcd.shape[0]
    center, scale = _frame(eval_pcd, normalize)
    pcd_t = torch.from_numpy(apply_frame(eval_pcd, center, scale)).unsqueeze(0).to(model.device)
    with torch.no_grad():
        w, m = model.predict(q, pcd_t)
    w = w[0].cpu().numpy()
    m_world = m[0].cpu().numpy()
    m_world[:, :3] = undo_frame(m_world[:, :3], center, scale)     # directions (if any) are unit vectors
    if use_argmax:
        target = m_world[w.argmax()]
    else:
        target = m_world[np.random.choice(len(w), p=w / (w.sum() + 1e-8))]
    pt_w = w.reshape(N, model.cfg.k_components).sum(-1)
    return target, w, m_world, pt_w / (pt_w.max() + 1e-8)


def adapt_from_history(model: SCOUT, eval_pcd: np.ndarray, hist_pcds: Sequence[np.ndarray],
                       hist_actions: Sequence[np.ndarray], hist_outcomes: Sequence[np.ndarray],
                       n_steps: int, normalize: bool) -> torch.Tensor:
    """``n_steps`` inner-loop updates of q0 on the failed attempts so far.

    All history is expressed in the *current* query cloud's frame, matching
    how the dataset normalises a training sample.  Actions are treated as
    positions in their first three dims; any direction dims are left as is.
    """
    if n_steps <= 0 or len(hist_pcds) == 0:
        return model.q0.detach().clone()
    center, scale = _frame(eval_pcd, normalize)

    def f_pcd(x):
        return apply_frame(x, center, scale)

    def f_act(a):
        a = np.asarray(a, dtype=np.float32).copy()
        a[:3] = apply_frame(a[:3], center, scale)
        return a

    dev = model.device
    hp = torch.from_numpy(np.stack([f_pcd(x) for x in hist_pcds])).to(dev)
    ha = torch.from_numpy(np.stack([f_act(a) for a in hist_actions])).to(dev)
    ho = torch.from_numpy(np.stack([f_pcd(x) for x in hist_outcomes])).to(dev)
    return model.adapt(hp, ha, ho, n_steps=n_steps)
