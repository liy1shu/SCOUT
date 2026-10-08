#!/usr/bin/env python
"""Split raw door pickles into the per-trajectory layout read by DoorTrajectoryDataset.

    python scripts/prepare_door_dataset.py --raw-dir data/door/raw_pkls --out-dir data/door/train_val

Writes ``<out-dir>/train/{i}.pkl`` for every trajectory in ``train_*.pkl`` and
``<out-dir>/val/{i}.pkl`` for every trajectory in ``test_*.pkl`` (the test
*objects* double as the validation set for model selection; closed-loop
evaluation runs on them in simulation), plus an ``indices.json`` per split.
Use ``--val-from-train <fraction>`` to hold out training trajectories instead.
"""

import argparse
import glob
import json
import os
import pickle as pkl

import numpy as np
from tqdm import tqdm


def _load_all(paths):
    trajs = []
    for path in paths:
        with open(path, "rb") as f:
            trajs.extend(pkl.load(f))
    return trajs


def _write_split(trajs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, traj in enumerate(tqdm(trajs, desc=os.path.basename(out_dir))):
        with open(os.path.join(out_dir, f"{i}.pkl"), "wb") as f:
            pkl.dump(traj, f)
    with open(os.path.join(out_dir, "indices.json"), "w") as f:
        json.dump({"data_idx": {str(i): i for i in range(len(trajs))}}, f)
    print(f"{out_dir}: {len(trajs)} trajectories")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-from-train", type=float, default=0.0,
                        help="hold out this fraction of train trajectories as val instead of using test_*.pkl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train = _load_all(sorted(glob.glob(os.path.join(args.raw_dir, "train_*.pkl"))))
    test = _load_all(sorted(glob.glob(os.path.join(args.raw_dir, "test_*.pkl"))))
    print(f"loaded {len(train)} train / {len(test)} test trajectories from {args.raw_dir}")

    if args.val_from_train > 0:
        perm = np.random.RandomState(args.seed).permutation(len(train))
        n_val = int(round(args.val_from_train * len(train)))
        val = [train[i] for i in perm[:n_val]]
        train = [train[i] for i in perm[n_val:]]
    else:
        val = test

    _write_split(train, os.path.join(args.out_dir, "train"))
    _write_split(val, os.path.join(args.out_dir, "val"))


if __name__ == "__main__":
    main()
