#!/usr/bin/env python
"""Meta-train SCOUT on one task.

    python scripts/train.py --task pick_bar --data data/pick_bar.pkl
    python scripts/train.py --task faucet   --data data/faucet.pkl --set train.epochs=500
    python scripts/train.py --task door     --data data/door/train_val --no-wandb

Checkpoints go to ``<save-dir>/<task>/{best,best_gap,latest}.pt`` and embed
the model config, so rollout.py does not need matching ``--set model.*`` flags.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Must precede the torch import (see scout/envs/metaworld/__init__.py). Expandable segments avoid the
# sporadic segfaults in the second-order backward (gradient checkpointing) seen with a fragmented cache.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from scout.cli import add_task_args, load_config  # noqa: E402
from scout.data import AttemptHistoryDataset, DoorTrajectoryDataset  # noqa: E402
from scout.models.scout import SCOUT  # noqa: E402
from scout.training import run_meta_training  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_task_args(parser)
    parser.add_argument("--data", required=True,
                        help="ISE: collected pickle. Door: directory with train/ and val/ (prepare_door_dataset.py)")
    parser.add_argument("--save-dir", default="ckpts")
    parser.add_argument("--run-name", default=None, help="checkpoint sub-directory and wandb run name (default: task)")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-project", default="scout")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args)

    if cfg.family == "metaworld":
        train_set, val_set = AttemptHistoryDataset.load_split(
            args.data, train_ratio=cfg.data.train_ratio, n_pts=cfg.data.n_pts,
            augment_prob=cfg.data.augment_prob, normalize=cfg.data.normalize, seed=cfg.data.split_seed,
            shuffle=cfg.data.shuffle_split)
    else:
        train_set = DoorTrajectoryDataset(args.data, "train", augment_prob=cfg.data.augment_prob)
        val_set = DoorTrajectoryDataset(args.data, "val", augment_prob=0.0)
    print(f"Train: {len(train_set)} samples | Val: {len(val_set)} samples")

    torch.manual_seed(cfg.train.seed)
    model = SCOUT(cfg.model).to(args.device)
    run_name = args.run_name or cfg.name
    run_meta_training(model, train_set, val_set, cfg,
                      save_dir=os.path.join(args.save_dir, run_name),
                      wandb_project=args.wandb_project, wandb_run=run_name,
                      wandb_entity=args.wandb_entity, use_wandb=not args.no_wandb,
                      resume=args.resume)


if __name__ == "__main__":
    main()
