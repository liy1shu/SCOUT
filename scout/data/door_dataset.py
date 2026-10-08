"""Dataset for the ambiguous-door benchmark.

Directory layout (produced by ``scripts/prepare_door_dataset.py``)::

    <root>/
      train/ indices.json  0.pkl 1.pkl ...
      val/   indices.json  0.pkl 1.pkl ...

``indices.json`` maps a sample index to a pickle index (``"data_idx"``) and,
optionally, to a trajectory index inside that pickle (``"traj_idx"``) when a
pickle holds a list of trajectories.  Each trajectory is a dict produced by
``scout.envs.door.collect.generate_trajectory``::

    pcds        [T, N, 3]  observed cloud before each interaction
    points      [T, 3]     contact point of each interaction
    directions  [T, 3]     unit pull direction of each interaction
    obs_flows   [T, N, 3]  per-point flow caused by each interaction
    scores      [T]        normalised joint progress of each interaction
    evaluate    dict with the query action(s) at the final state

The last interaction (index T-1) is the ground-truth query; the first T-1 are
the failed history.  Point clouds are used in raw world coordinates.

Each item yields::

    eval_pcd      [N, 3]      eval_action  [6]        eval_outcome  [N, 3]  (flow)
    hist_pcds     [H, N, 3]   hist_actions [H, 6]     hist_outcomes [H, N, 3]  (flows)
"""

from __future__ import annotations

import json
import os
import pickle as pkl

import numpy as np
import torch
from torch.utils.data import Dataset


class DoorTrajectoryDataset(Dataset):
    def __init__(self, root_dir: str, subset: str, augment_prob: float = 0.0):
        self.directory = os.path.join(root_dir, subset)
        with open(os.path.join(self.directory, "indices.json")) as f:
            indices = json.load(f)
        self.augment_prob = augment_prob

        data_idx = indices["data_idx"]
        self.traj_idx = indices.get("traj_idx")
        # Deduplicate samples that point to the same trajectory.
        seen, unique = set(), {}
        for sample_idx, d in data_idx.items():
            key = (d, None if self.traj_idx is None else self.traj_idx[sample_idx])
            if key not in seen:
                seen.add(key)
                unique[sample_idx] = d
        self.sample_to_data = unique
        self.sample_ids = list(unique.keys())

    def __len__(self):
        return len(self.sample_ids)

    def _load(self, idx):
        sample_idx = self.sample_ids[idx]
        with open(os.path.join(self.directory, f"{self.sample_to_data[sample_idx]}.pkl"), "rb") as f:
            traj = pkl.load(f)
        if self.traj_idx is not None:
            traj = traj[self.traj_idx[sample_idx]]
        return traj

    def __getitem__(self, idx):
        traj = self._load(idx)
        gt = len(traj["pcds"]) - 1

        eval_pcd = traj["evaluate"]["pcds"][gt]
        eval_action = np.concatenate([traj["evaluate"]["points"][gt],
                                      traj["evaluate"]["directions"][gt]], axis=-1)
        eval_flow = traj["obs_flows"][-1]

        hist_pcds = traj["pcds"][:-1]
        hist_actions = np.concatenate([traj["points"][:-1], traj["directions"][:-1]], axis=-1)
        hist_flows = traj["obs_flows"][:-1]

        if self.augment_prob > 0 and np.random.rand() < self.augment_prob:
            hist_pcds = np.concatenate([hist_pcds, eval_pcd[None]], axis=0)
            hist_actions = np.concatenate([hist_actions, eval_action[None]], axis=0)
            hist_flows = np.concatenate([hist_flows, eval_flow[None]], axis=0)

        f32 = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        return {
            "eval_pcd": f32(eval_pcd),
            "eval_action": f32(eval_action),
            "eval_outcome": f32(eval_flow),
            "hist_pcds": f32(hist_pcds),
            "hist_actions": f32(hist_actions),
            "hist_outcomes": f32(hist_flows),
        }
