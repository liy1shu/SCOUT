"""Dataset for the ISE / Meta-World tasks (pick bar, push bar, turn faucet, open box).

Input format
------------
A pickle holding a list of *samples* as written by
``scout.envs.metaworld.collect``.  One sample is one episode of the scripted
expert trying wrong actions before the correct one::

    {
      "observation":        [pcd_0, ..., pcd_{H-1}]   object cloud before each failed attempt
      "history_actions":    [a_0, ..., a_{H-1}]       3-D target of each failed attempt
      "history_outcomes":   [pcd'_0, ..., pcd'_{H-1}] object cloud after each failed attempt
      "ground_truth_action":  a*                      target of the final, successful attempt
      "ground_truth_outcome": pcd'*                   object cloud after the successful attempt
      "ground_truth_observation": pcd*   (optional)   object cloud before the successful attempt
      "num_history": H, "steps": ..., "env_name": ...
    }

Each item yields a dict of tensors in one normalised frame (see
:func:`scout.data.pointcloud.normalize_pcd`)::

    eval_pcd      [N, 3]   query observation
    eval_action   [3]      ground-truth query action
    eval_outcome  [N, 3]   post-action cloud of the query (unused by the ISE loss)
    hist_pcds     [H, N, 3]
    hist_actions  [H, 3]
    hist_outcomes [H, N, 3]

Processing matches the paper experiments: episodes without history are
dropped, the train/val split is sequential by default (first ``train_ratio`` of the pickle is training data), every
cloud is sub-sampled to ``n_pts`` points, and query + history share the
frame of the query cloud.

Query observation: the collector stores ``ground_truth_observation`` (the
cloud before the successful attempt).  The object is reset to the same pose
before every attempt, so for pickles without it the last history observation
(the same cloud up to rendering noise) is used instead.
"""

from __future__ import annotations

import pickle
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from scout.data.pointcloud import apply_frame, normalize_pcd, sample_pcd


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Pickles written with numpy 2.x reference ``numpy._core``; remap for numpy 1.x."""

    def find_class(self, module, name):
        return super().find_class(module.replace("numpy._core", "numpy.core"), name)


class AttemptHistoryDataset(Dataset):
    def __init__(self, samples: Sequence[dict], n_pts: int = 1200,
                 augment_prob: float = 0.0, normalize: bool = True, min_history: int = 1):
        self.n_pts = n_pts
        self.augment_prob = augment_prob
        self.normalize = normalize
        self.samples = [s for s in samples
                        if self._query_pcd(s) is not None and s.get("num_history", len(s["observation"])) >= min_history]
        dropped = len(samples) - len(self.samples)
        if dropped:
            print(f"[AttemptHistoryDataset] skipped {dropped} episodes (no history or no query cloud)")

    # ── construction helpers ─────────────────────────────────────────────────
    @staticmethod
    def _query_pcd(sample: dict):
        if sample.get("ground_truth_observation") is not None:
            return sample["ground_truth_observation"]
        if len(sample.get("observation", [])) > 0:
            return sample["observation"][-1]
        return None

    @classmethod
    def load(cls, path: str) -> List[dict]:
        with open(path, "rb") as f:
            return _NumpyCompatUnpickler(f).load()

    @classmethod
    def load_split(cls, path: str, train_ratio: float = 0.9, n_pts: int = 1200,
                   augment_prob: float = 0.1, normalize: bool = True, seed: int = 0,
                   shuffle: bool = False) -> Tuple["AttemptHistoryDataset", "AttemptHistoryDataset"]:
        """Train/val split of a single collected pickle.

        Episodes without history are dropped first.  The default split is
        sequential (first ``train_ratio`` -> train), as in the original
        experiments; ``shuffle=True`` uses a seeded permutation instead.  If
        the pickle is too small for a validation split, validation reuses
        the training episodes.
        """
        samples = [s for s in cls.load(path) if s.get("num_history", len(s["observation"])) > 0]
        order = np.random.RandomState(seed).permutation(len(samples)) if shuffle else np.arange(len(samples))
        n_train = max(1, int(len(samples) * train_ratio))
        train = [samples[i] for i in order[:n_train]]
        val = [samples[i] for i in order[n_train:]] or train
        return (cls(train, n_pts=n_pts, augment_prob=augment_prob, normalize=normalize),
                cls(val, n_pts=n_pts, augment_prob=0.0, normalize=normalize))

    # ── Dataset API ──────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        eval_pcd = sample_pcd(self._query_pcd(s), self.n_pts)
        eval_action = np.asarray(s["ground_truth_action"], dtype=np.float32)
        eval_outcome = sample_pcd(s["ground_truth_outcome"], self.n_pts)

        hist_pcds = [sample_pcd(o, self.n_pts) for o in s["observation"]]
        hist_actions = [np.asarray(a, dtype=np.float32) for a in s["history_actions"]]
        hist_outcomes = [sample_pcd(o, self.n_pts) for o in s["history_outcomes"]]

        # Augmentation: pretend the successful attempt was also observed.
        if self.augment_prob > 0 and np.random.rand() < self.augment_prob:
            hist_pcds.append(eval_pcd.copy())
            hist_actions.append(eval_action.copy())
            hist_outcomes.append(eval_outcome.copy())

        H = len(hist_pcds)
        hist_pcds = np.stack(hist_pcds) if H else np.zeros((0, self.n_pts, 3), np.float32)
        hist_actions = np.stack(hist_actions) if H else np.zeros((0, 3), np.float32)
        hist_outcomes = np.stack(hist_outcomes) if H else np.zeros((0, self.n_pts, 3), np.float32)

        if self.normalize:
            _, center, scale = normalize_pcd(eval_pcd)
            eval_pcd = apply_frame(eval_pcd, center, scale)
            eval_action = apply_frame(eval_action, center, scale)
            eval_outcome = apply_frame(eval_outcome, center, scale)
            hist_pcds = apply_frame(hist_pcds, center, scale)
            hist_actions = apply_frame(hist_actions, center, scale)
            hist_outcomes = apply_frame(hist_outcomes, center, scale)

        return {
            "eval_pcd": torch.from_numpy(np.ascontiguousarray(eval_pcd, dtype=np.float32)),
            "eval_action": torch.from_numpy(np.ascontiguousarray(eval_action, dtype=np.float32)),
            "eval_outcome": torch.from_numpy(np.ascontiguousarray(eval_outcome, dtype=np.float32)),
            "hist_pcds": torch.from_numpy(np.ascontiguousarray(hist_pcds, dtype=np.float32)),
            "hist_actions": torch.from_numpy(np.ascontiguousarray(hist_actions, dtype=np.float32)),
            "hist_outcomes": torch.from_numpy(np.ascontiguousarray(hist_outcomes, dtype=np.float32)),
        }
