#!/usr/bin/env python
"""Split raw door pickles into the per-trajectory layout read by DoorTrajectoryDataset.

    python scripts/prepare_door_dataset.py --raw-dir data/door/raw_pkls --out-dir data/door/train_val

Writes ``<out-dir>/{train,val}/{i}.pkl`` (one trajectory per file) plus an
``indices.json`` per split.  A random ``--val-fraction`` (default 20 %) of the
``train_*.pkl`` trajectories is held out as the validation set for model
selection, as for the released checkpoint.
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
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="fraction of train trajectories held out as val")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train = _load_all(sorted(glob.glob(os.path.join(args.raw_dir, "train_*.pkl"))))
    print(f"loaded {len(train)} train trajectories from {args.raw_dir}")

    perm = np.random.RandomState(args.seed).permutation(len(train))
    n_val = int(round(args.val_fraction * len(train)))
    val = [train[i] for i in perm[:n_val]]
    train = [train[i] for i in perm[n_val:]]

    _write_split(train, os.path.join(args.out_dir, "train"))
    _write_split(val, os.path.join(args.out_dir, "val"))


if __name__ == "__main__":
    main()
