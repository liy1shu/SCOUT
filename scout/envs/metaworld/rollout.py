"""Closed-loop evaluation of SCOUT on the ISE tasks.

Per episode: observe the object -> (adapt q from the failed attempts so far)
-> predict a target -> execute it with the task's evaluation policy -> record
the outcome cloud -> reset the object -> repeat until success or the attempt
budget is exhausted.  The reported metric is attempts-to-success with failed
episodes counted as ``max_attempts``.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional

import mujoco
import numpy as np
import torch
from tqdm import tqdm

from scout.configs import RolloutConfig
from scout.data.pointcloud import sample_pcd
from scout.envs.metaworld import viz
from scout.envs.metaworld.pointcloud import FRAME_H, FRAME_W, camera_params, close_renderer
from scout.envs.metaworld.tasks.base import MetaWorldTask
from scout.inference import adapt_from_history, predict_target
from scout.models.scout import SCOUT


# ── one episode ──────────────────────────────────────────────────────────────
def run_episode(task: MetaWorldTask, env_name: str, model: SCOUT, cfg: RolloutConfig,
                n_pts: int, normalize: bool, use_argmax: bool,
                video_dir: Optional[str] = None, ep_idx: int = 0, mode_str: str = "TTT") -> dict:
    env, base_env = task.make_env(env_name, for_rollout=True)
    renderer = mujoco.Renderer(base_env.model, FRAME_H, FRAME_W)

    obs, _ = env.reset()
    task.after_reset(base_env, cfg.max_steps_per_attempt, for_rollout=True)
    canonical = task.canonical_observation(env, base_env, renderer)
    if canonical is not None:
        obs = base_env._get_obs()
    gt_target = task.gt_target(obs, base_env)
    policy = task.make_eval_policy(env_name, base_env)

    hist_pcds, hist_actions, hist_outcomes = [], [], []
    hist_w, hist_m = [], []
    last_outcome = None
    attempts: List[dict] = []
    per_attempt = []
    frames = []
    record = video_dir is not None

    for attempt in range(cfg.max_attempts):
        pts = canonical[0] if canonical is not None else task.object_cloud(env, renderer)[0]
        eval_pcd = sample_pcd(pts, n_pts)

        q = adapt_from_history(model, eval_pcd, hist_pcds, hist_actions, hist_outcomes,
                               n_steps=cfg.ttt_steps, normalize=normalize)
        target, w_nk, m_nk, pt_w = predict_target(model, q, eval_pcd, use_argmax, normalize)

        if record:
            cam = camera_params(base_env)
            panel = viz.make_weight_panel(task, eval_pcd, w_nk, m_nk, target, hist_w, hist_m,
                                          hist_actions, last_outcome, attempt, cfg.max_attempts,
                                          f"{mode_str} {env_name}", gt_target=gt_target)

        policy = task.begin_attempt(policy, env_name, base_env, target)
        success, outcome_pcd = False, None

        for _ in range(cfg.max_steps_per_attempt):
            a = policy.get_adaptive_action(obs)
            if policy.current_is_keyframe and policy.keyframe_type == "lift":
                outcome_pcd = sample_pcd(task.object_cloud(env, renderer)[0], n_pts)

            if record:
                text = [f"Ep {ep_idx + 1}  [{mode_str}]  {env_name}",
                        f"Attempt {attempt + 1}/{cfg.max_attempts}  |  {task.phase_name(policy)}",
                        f"Target ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})"]
                frame = viz.draw_env_frame(renderer, base_env, w_nk, m_nk, target, hist_actions,
                                           cam, text, gt_target=gt_target)
                frames.append(np.concatenate([viz.draw_attempt_dots(frame, per_attempt), panel], axis=1))

            obs, _, terminated, truncated, info = env.step(a)
            if int(info.get("success", 0)) == 1:
                success = True
                break
            if policy.attempt_done:
                break
            stop, reset_env = task.handle_termination(terminated, truncated)
            if stop:
                if reset_env:
                    obs, _ = env.reset()
                break

        if outcome_pcd is None:   # 'lift' keyframe never fired: use the current state
            outcome_pcd = sample_pcd(task.object_cloud(env, renderer)[0], n_pts)

        per_attempt.append(success)
        attempts.append({"attempt_idx": attempt, "eval_pcd": eval_pcd.copy(),
                         "weights_nk": w_nk.copy(), "means_nk": m_nk.copy(),
                         "point_weights": pt_w.copy(), "predicted_target": target.copy(),
                         "outcome_pcd": outcome_pcd.copy(), "success": success,
                         "q": q.cpu().numpy().copy()})
        if success:
            break

        hist_pcds.append(eval_pcd.copy())
        hist_actions.append(target.copy())
        hist_outcomes.append(outcome_pcd.copy())
        hist_w.append(w_nk.copy())
        hist_m.append(m_nk.copy())
        last_outcome = outcome_pcd
        obs = task.reset_object(base_env, env_name)

    close_renderer(renderer)
    env.close()

    ep_success = any(a["success"] for a in attempts)
    if record and frames:
        tag = "success" if ep_success else f"fail_{len(attempts)}att"
        viz.save_video(frames, os.path.join(video_dir, f"ep{ep_idx + 1:03d}_{env_name}_{tag}_{mode_str}.mp4"))

    return {"env_name": env_name, "success": ep_success, "n_attempts": len(attempts),
            "per_attempt_success": per_attempt, "attempts": attempts}


# ── full evaluation ──────────────────────────────────────────────────────────
def run_rollouts(task: MetaWorldTask, model: SCOUT, cfg: RolloutConfig, n_pts: int,
                 normalize: bool, out_dir: str, video_dir: Optional[str] = None) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    use_ttt = cfg.ttt_steps > 0
    if cfg.proposal_selection == "auto":
        use_argmax = use_ttt
    else:
        use_argmax = cfg.proposal_selection == "argmax"
    mode_str = f"TTT-{cfg.ttt_steps}step" if use_ttt else "No-TTT"
    print(f"Rollout [{task.name}] {mode_str} | {cfg.n_episodes} episodes x {cfg.max_attempts} attempts "
          f"| proposal={'argmax' if use_argmax else 'sample'}")

    episodes = []
    for ep in tqdm(range(cfg.n_episodes), desc=f"Rollout [{mode_str}]"):
        env_name = str(np.random.choice(list(task.env_names)))
        ep_data = run_episode(task, env_name, model, cfg, n_pts, normalize, use_argmax,
                              video_dir=video_dir, ep_idx=ep, mode_str=mode_str)
        episodes.append(ep_data)
        print(f"  Ep {ep + 1} [{env_name}]: {'ok' if ep_data['success'] else 'FAIL'}  "
              f"{ep_data['n_attempts']} attempts  {ep_data['per_attempt_success']}")

    combined = [e["n_attempts"] if e["success"] else cfg.max_attempts for e in episodes]
    summary = {
        "task": task.name, "mode": mode_str, "ttt_steps": cfg.ttt_steps,
        "n_episodes": cfg.n_episodes, "max_attempts": cfg.max_attempts,
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "mean_attempts": float(np.mean(combined)),
        "std_attempts": float(np.std(combined)),
        "sem_attempts": float(np.std(combined) / np.sqrt(max(len(combined), 1))),
        "combined_attempts": combined,
    }
    for name in task.env_names:
        eps = [e for e in episodes if e["env_name"] == name]
        if eps:
            summary[f"success_rate/{name}"] = float(np.mean([e["success"] for e in eps]))
    ok = [e for e in episodes if e["success"]]
    if ok:
        summary["mean_attempts_successful_only"] = float(np.mean([e["n_attempts"] for e in ok]))

    tag = f"{task.name}_ttt{cfg.ttt_steps}"
    with open(os.path.join(out_dir, f"episodes_{tag}.pkl"), "wb") as f:
        pickle.dump({"summary": summary, "episodes": episodes}, f)
    with open(os.path.join(out_dir, f"summary_{tag}.pkl"), "wb") as f:
        pickle.dump(summary, f)

    print(f"\n[{task.name} | {mode_str}] success rate {summary['success_rate']:.3f} | "
          f"attempts (fail -> {cfg.max_attempts}): {summary['mean_attempts']:.3f} "
          f"+/- {summary['sem_attempts']:.3f} (s.e.m.)")
    for name in task.env_names:
        if f"success_rate/{name}" in summary:
            print(f"  {name}: SR {summary[f'success_rate/{name}']:.3f}")
    print(f"Saved -> {out_dir}/episodes_{tag}.pkl, summary_{tag}.pkl")
    return summary
