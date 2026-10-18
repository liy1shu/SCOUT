#!/usr/bin/env python
"""Evaluate a trained SCOUT checkpoint in the simulator.

ISE / Meta-World (attempts-to-success over N episodes):
    python scripts/rollout.py --task pick_bar --ckpt ckpts/pick_bar/best.pt --ttt-steps 1
    python scripts/rollout.py --task pick_bar --ckpt ckpts/pick_bar/best.pt --ttt-steps 0   # no adaptation
    python scripts/rollout.py --task faucet   --ckpt ckpts/faucet/latest.pt --no-video

Ambiguous doors (success rate / steps over the test objects, sweep over k):
    python scripts/rollout.py --task door --ckpt ckpts/door/best.pt --ttt-steps 5 1 0 \
        --pm-dir ~/datasets/failure_history_door/raw --metadata metadata/multimodal_door.json

Checkpoints written by scripts/train.py carry their own model config; for the
released weights (no embedded config) the task preset supplies it.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import torch  # noqa: E402

from scout.cli import add_task_args, load_config  # noqa: E402
from scout.models.scout import SCOUT  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_task_args(parser)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--ttt-steps", type=int, nargs="+", default=None,
                        help="inner-loop steps at test time; several values run a sweep (default: task preset)")
    parser.add_argument("--n-episodes", type=int, default=None, help="(ISE only) default: task preset")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--video-dir", default=None, help="directory for per-episode mp4s (default: <out-dir>/<task>/videos_ttt<k>)")
    parser.add_argument("--no-video", action="store_true", help="disable video recording (use for full sweeps)")
    parser.add_argument("--seed-episodes", action="store_true",
                        help="(ISE only) reproducible episodes: reseed numpy/torch with seed_base + i before "
                             "episode i and seed the env with i (base: --set rollout.seed_base=..., default 10000)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Door
    parser.add_argument("--pm-dir", default="~/datasets/failure_history_door/raw")
    parser.add_argument("--metadata", default="metadata/multimodal_door.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-objects", type=int, default=None, help="(door) only roll out the first N objects")
    args = parser.parse_args()
    cfg = load_config(args)

    model = SCOUT.from_checkpoint(args.ckpt, cfg=cfg.model, device=args.device, prefer_embedded=True)
    ttt_steps_list = args.ttt_steps or [cfg.rollout.ttt_steps]
    out_dir = os.path.join(args.out_dir, cfg.name)

    for k in ttt_steps_list:
        rcfg = cfg.override(**{"rollout.ttt_steps": k, "rollout.n_episodes": args.n_episodes,
                               "rollout.seed_episodes": True if args.seed_episodes else None}).rollout
        video_dir = None if args.no_video else (args.video_dir or os.path.join(out_dir, f"videos_ttt{k}"))
        if cfg.family == "metaworld":
            from scout.envs.metaworld import get_task
            from scout.envs.metaworld.rollout import run_rollouts
            run_rollouts(get_task(cfg.name), model, rcfg, n_pts=cfg.data.n_pts,
                         normalize=cfg.data.normalize, out_dir=out_dir, video_dir=video_dir)
        else:
            from scout.envs.door.rollout import run_door_rollouts
            run_door_rollouts(model, rcfg, pm_dir=os.path.expanduser(args.pm_dir),
                              metadata_path=args.metadata, out_dir=out_dir, split=args.split,
                              max_objects=args.max_objects, video_dir=video_dir)


if __name__ == "__main__":
    main()
