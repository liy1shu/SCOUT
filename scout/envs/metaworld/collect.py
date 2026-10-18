"""Offline data collection for the ISE tasks.

The scripted expert tries a sequence of wrong targets before the correct one
(``policy.get_adaptive_action``).  Around every attempt we record the object
cloud before the attempt (at the ``hover`` keyframe), the executed target, and
the object cloud after the attempt (at the ``lift`` keyframe).  Episodes that
do not end in success are discarded.
"""

from __future__ import annotations

import os
import pickle
from typing import List, Optional

import mujoco
from tqdm import tqdm

from scout.envs.metaworld.pointcloud import FRAME_H, FRAME_W, close_renderer
from scout.envs.metaworld.tasks.base import MetaWorldTask


def collect_episode(task: MetaWorldTask, env_name: str, max_steps: int = 50000) -> Optional[dict]:
    env, base_env = task.make_env(env_name, for_rollout=False)
    policy = task.make_expert_policy(env_name, base_env)

    obs, info = env.reset()
    task.after_reset(base_env, max_steps, for_rollout=False)
    renderer = mujoco.Renderer(base_env.model, FRAME_H, FRAME_W)

    canonical = task.canonical_observation(env, base_env, renderer)
    if canonical is not None:
        obs = base_env._get_obs()

    attempts: List[dict] = []
    pending = None
    done, step, success = False, 0, False

    while not done and step < max_steps and task.collection_may_continue(policy):
        a = policy.get_adaptive_action(obs)

        if policy.current_is_keyframe:
            if policy.keyframe_type == "hover":
                pts, clr = canonical if canonical is not None else task.object_cloud(env, renderer)
                pending = {"input_pc": pts, "input_colors": clr, "target": policy.target.copy()}
            elif policy.keyframe_type == "lift" and pending is not None:
                pts, clr = task.object_cloud(env, renderer)
                pending["outcome_pc"], pending["outcome_colors"] = pts, clr
                attempts.append(pending)
                pending = None

        obs, _, terminated, truncated, info = env.step(a)
        step += 1

        if policy.attempt_done:
            policy.attempt_done = False
            obs = task.reset_object(base_env, env_name)

        success = int(info["success"]) == 1 and task.counts_as_success(policy)
        done = success or (task.stop_on_termination() and (terminated or truncated))

    if success and pending is not None:            # outcome of the final, successful attempt
        pts, clr = task.object_cloud(env, renderer)
        pending["outcome_pc"], pending["outcome_colors"] = pts, clr
        attempts.append(pending)

    close_renderer(renderer)
    env.close()

    if not success or len(attempts) == 0:
        return None

    hist, final = attempts[:-1], attempts[-1]
    return {
        "observation": [a["input_pc"] for a in hist],
        "observation_colors": [a["input_colors"] for a in hist],
        "history_actions": [a["target"] for a in hist],
        "history_outcomes": [a["outcome_pc"] for a in hist],
        "history_outcome_colors": [a["outcome_colors"] for a in hist],
        "ground_truth_observation": final["input_pc"],
        "ground_truth_action": final["target"],
        "ground_truth_outcome": final["outcome_pc"],
        "ground_truth_outcome_colors": final["outcome_colors"],
        "num_history": len(hist),
        "steps": step,
        "env_name": env_name,
    }


def collect_dataset(task: MetaWorldTask, num_episodes: int, output_path: str,
                    max_steps: int = 50000) -> List[dict]:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    dataset: List[dict] = []
    tried = 0
    with tqdm(total=num_episodes) as pbar:
        while len(dataset) < num_episodes:
            tried += 1
            env_name = task.choose_env_name(dataset)
            pbar.set_description(f"[{task.name}] episode {len(dataset) + 1}/{num_episodes} ({env_name})")
            sample = collect_episode(task, env_name, max_steps=max_steps)
            if sample is None:
                pbar.write(f"failed ({env_name}), skipped")
                continue
            dataset.append(sample)
            pbar.write(f"ok  ({env_name}, history={sample['num_history']}, steps={sample['steps']})")
            pbar.update(1)
            with open(output_path, "wb") as f:
                pickle.dump(dataset, f)

    print(f"\nSaved {len(dataset)} samples ({tried} episodes run) to {output_path} "
          f"({os.path.getsize(output_path) / 1e6:.1f} MB)")
    return dataset
