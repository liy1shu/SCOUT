#!/usr/bin/env python
"""Collect offline meta-training data.

ISE / Meta-World tasks (one pickle of episodes):
    python scripts/collect_data.py --task pick_bar --num-episodes 200 --output data/pick_bar.pkl
    python scripts/collect_data.py --task push_bar --num-episodes 200 --output data/push_bar.pkl
    python scripts/collect_data.py --task faucet   --num-episodes 200 --output data/faucet.pkl
    python scripts/collect_data.py --task box      --num-episodes 500 --output data/box.pkl

Ambiguous doors (raw per-worker pickles; run prepare_door_dataset.py afterwards):
    python scripts/collect_data.py --task door --pm-dir ~/datasets/failure_history_door/raw \
        --metadata metadata/multimodal_door.json --output-dir data/door
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scout.cli import add_task_args, load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_task_args(parser)
    parser.add_argument("--num-episodes", type=int, default=200, help="(ISE) successful episodes to keep")
    parser.add_argument("--output", type=str, default=None, help="(ISE) output pickle, default data/<task>.pkl")
    parser.add_argument("--max-steps", type=int, default=50000, help="(ISE) env steps per episode")
    # Door
    parser.add_argument("--pm-dir", type=str, default="~/datasets/failure_history_door/raw",
                        help="(door) PartNet-Mobility multimodal door assets")
    parser.add_argument("--metadata", type=str, default="metadata/multimodal_door.json",
                        help="(door) split json with 'train-train' / 'test' object ids")
    parser.add_argument("--output-dir", type=str, default="data/door")
    parser.add_argument("--num-processes", type=int, default=16)
    parser.add_argument("--traj-per-obj-train", type=int, default=100)
    parser.add_argument("--traj-per-obj-test", type=int, default=20)
    parser.add_argument("--max-trials-train", type=int, default=200)
    parser.add_argument("--max-trials-test", type=int, default=50)
    parser.add_argument("--random-action-num", type=int, default=10, help="(door) query actions per trajectory")
    parser.add_argument("--history-len", type=int, default=-1, help="(door) H wrong attempts; -1 = random in [1, 8)")
    parser.add_argument("--max-objects", type=int, default=None, help="(door) only use the first N objects of each split (smoke tests)")
    args = parser.parse_args()
    cfg = load_config(args)

    if cfg.family == "metaworld":
        from scout.envs.metaworld import get_task
        from scout.envs.metaworld.collect import collect_dataset
        task = get_task(cfg.name)
        output = args.output or f"data/{cfg.name}.pkl"
        collect_dataset(task, args.num_episodes, output, max_steps=args.max_steps)
    else:
        from scout.envs.door.collect import collect_door_dataset
        collect_door_dataset(
            pm_dir=os.path.expanduser(args.pm_dir), metadata_path=args.metadata,
            output_dir=args.output_dir, num_processes=args.num_processes,
            traj_per_obj_train=args.traj_per_obj_train, traj_per_obj_test=args.traj_per_obj_test,
            max_trials_train=args.max_trials_train, max_trials_test=args.max_trials_test,
            evaluate_random_action_cnt=args.random_action_num, history_len=args.history_len,
            max_objects=args.max_objects)


if __name__ == "__main__":
    main()
