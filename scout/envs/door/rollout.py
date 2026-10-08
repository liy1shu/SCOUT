"""Closed-loop evaluation on the ambiguous doors.

At every step the policy proposes (contact point, pull direction) candidates
ranked by mixture weight; the suction gripper is teleported to the first
candidate it can attach to, pulls for 100 simulation steps, and the observed
per-point flow of the door is added to the history.  An episode succeeds when
the target joint passes ``success_threshold`` of its range.
"""

from __future__ import annotations

import json
import os
import pickle as pkl
from typing import Dict, List, Optional

import numpy as np
import pybullet as p
import torch

from scout.configs import RolloutConfig
from scout.envs.door.articulate import articulate_joint
from scout.envs.door.collect import N_PTS, _make_env, parse_obj_id
from scout.envs.door.partnet_mobility import PMObject
from scout.inference import adapt_from_history, predict_target
from scout.models.scout import SCOUT


def _closest_points(pcd: np.ndarray, targets: np.ndarray) -> np.ndarray:
    d = ((targets[:, None, :] - pcd[None, :, :]) ** 2).sum(-1)
    return pcd[d.argmin(axis=1)]


def rollout_door(model: SCOUT, cfg: RolloutConfig, pm_dir: str, door_id: str,
                 target_link: str, mode_id: int, video_path: Optional[str] = None) -> Optional[tuple]:
    obj_id = f"{door_id}_{target_link}_{mode_id + 1}"
    raw_data = PMObject(os.path.join(pm_dir, obj_id))
    available_joints = [j.name for j in raw_data.semantics.by_type("hinge") + raw_data.semantics.by_type("slider")]

    env = _make_env(pm_dir, obj_id, target_link, raw_data, camera_pos=[-3, 0, 2], camera_pos_random=False)
    # Note: the range is taken from the *last* movable joint listed in semantics.txt, as in
    # the evaluation used for the paper; the generated door assets freeze every other joint.
    init_angle = target_angle = 0.0
    for joint in available_joints:
        info = p.getJointInfo(env.render_env.obj_id, env.render_env.link_name_to_index[joint], env.render_env.client_id)
        init_angle, target_angle = info[8], info[9]
        env.set_joint_state(joint, init_angle)
    if target_angle == init_angle:
        print("No movement joint!")
        p.disconnect(env.render_env.client_id)
        return None

    recorder = None
    if video_path is not None:
        from scout.envs.door.viz import DoorRecorder
        recorder = DoorRecorder(env)

    def finish(step_id, success, angle, logs):
        if recorder is not None:
            recorder.set_overlay([f"{obj_id}  {'SUCCESS' if success else 'FAIL'} after {step_id + 1} steps",
                                  f"rel angle {angle:.2f}"], history=hist_actions)
            for _ in range(10):
                recorder.capture()
            recorder.save(video_path)
        p.disconnect(env.render_env.client_id)
        return step_id, success, angle, logs

    pcds, actions, w_log, pos_log, dir_log = [], [], [], [], []
    hist_pcds, hist_actions, hist_flows = [], [], []
    curr_rel_angle = 0.0
    logs: Dict = {}

    for step_id in range(cfg.max_attempts):
        curr_angle = env.get_joint_value(target_link)
        curr_rel_angle = (curr_angle - init_angle) / (target_angle - init_angle)
        curr_joint_angles = env.render_env.get_joint_angles()

        _, _, _, _, P_world, pc_seg, _ = env.render(filter_nonobj_pts=True, n_pts=N_PTS)

        # ── adapt + predict (raw world coordinates, no normalisation) ─────────
        q = adapt_from_history(model, P_world, hist_pcds, hist_actions, hist_flows,
                               n_steps=cfg.ttt_steps, normalize=False)
        _, weights, means, _ = predict_target(model, q, P_world, use_argmax=True, normalize=False)
        pos_means, dir_means = means[:, 0:3], means[:, 3:6]
        w_log.append(weights); pos_log.append(pos_means); dir_log.append(dir_means)

        # ── action selection: try candidates in decreasing weight ────────────
        order = np.argsort(weights)[::-1]
        best_points, best_flows = pos_means[order], dir_means[order]
        pcds.append(P_world)
        link_ixs = pc_seg == env.render_env.link_name_to_index[target_link]
        if not link_ixs.any():
            link_ixs = np.ones_like(link_ixs, dtype=bool)
        if cfg.closest_point:
            best_points = _closest_points(P_world[link_ixs], best_points)
        print(f"  step {step_id} | rel angle {curr_rel_angle:.3f}")
        if recorder is not None:
            recorder.set_overlay([f"{obj_id}  step {step_id + 1}/{cfg.max_attempts}",
                                  f"rel angle {curr_rel_angle:.2f} | TTT steps {cfg.ttt_steps} | history {len(hist_pcds)}"],
                                 point=best_points[0], direction=best_flows[0], history=hist_actions)
            for _ in range(5):
                recorder.capture()

        env.reset_gripper(target_link)
        p.stepSimulation(env.render_env.client_id)
        best_ix, contact = env.teleport(best_points, best_flows, target_link=target_link)
        if not contact:
            print("  cannot contact")
            continue
        best_flow, best_point = best_flows[best_ix], best_points[best_ix]
        if recorder is not None:
            recorder.set_overlay(recorder.text, point=best_point, direction=best_flow)
        env.attach()
        curr_action = np.concatenate([best_point, best_flow])
        actions.append(curr_action)

        env.pull_with_constraint(best_flow, target_link=target_link, n_steps=100, constraint=True)
        new_angle = env.get_joint_value(target_link)
        new_rel_angle = (new_angle - init_angle) / (target_angle - init_angle)

        # Analytical observed flow of the door link.
        P_world_new = articulate_joint(raw_data, curr_joint_angles, target_link, new_angle - curr_angle,
                                           P_world, pc_seg, env.render_env.link_name_to_index,
                                           env.render_env.T_world_base)
        obs_flow = P_world_new - P_world

        logs = {"pcds": pcds, "actions": actions, "gmm_weights": w_log, "gmm_pos": pos_log, "gmm_dir": dir_log,
                "history_pcds": list(hist_pcds), "history_actions": list(hist_actions),
                "history_flows": list(hist_flows)}

        if new_rel_angle > cfg.success_threshold:
            print("  succeed")
            return finish(step_id, True, new_rel_angle, logs)

        hist_pcds.append(P_world); hist_actions.append(curr_action); hist_flows.append(obs_flow)
        curr_rel_angle = new_rel_angle

    return finish(step_id, False, curr_rel_angle, logs)


def run_door_rollouts(model: SCOUT, cfg: RolloutConfig, pm_dir: str, metadata_path: str,
                      out_dir: str, split: str = "test", obj_ids: Optional[List[str]] = None,
                      max_objects: Optional[int] = None, video_dir: Optional[str] = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)
    if obj_ids is None:
        with open(metadata_path) as f:
            obj_ids = json.load(f)[split]
    obj_ids = obj_ids[:max_objects]
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    tag = f"door_{split}_ttt{cfg.ttt_steps}{'_cp' if cfg.closest_point else ''}"
    step_ids, srs, angles, all_logs = [], [], [], []

    def save() -> dict:
        mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
        summary = {"sr": mean(srs), "avg_steps": mean(step_ids), "avg_final_angle": mean(angles),
                   "n": len(srs), "ttt_steps": cfg.ttt_steps}
        with open(os.path.join(out_dir, f"{tag}.pkl"), "wb") as f:
            pkl.dump({"step_ids": step_ids, "srs": srs, "angles": angles, "all_logs": all_logs,
                      "metrics": summary}, f)
        return summary

    for obj in obj_ids:
        door_id, target_link, mode_id = parse_obj_id(obj)
        for trial in range(cfg.trials_per_object):
            video_path = None if video_dir is None else os.path.join(video_dir, f"{obj}_trial{trial}_ttt{cfg.ttt_steps}.mp4")
            res = rollout_door(model, cfg, pm_dir, door_id, target_link, mode_id, video_path=video_path)
            if res is None:
                break
            step_id, success, angle, logs = res
            logs.update(obj=obj, door_id=door_id, trial_id=trial)
            all_logs.append(logs)
            step_ids.append(step_id); srs.append(int(success)); angles.append(angle)
            print(f"{obj} trial {trial} | steps {step_id} | success {success} | final angle {angle:.3f}")

        save()   # rewrite after every object so partial sweeps are kept

    summary = save()
    print(f"\n[door | ttt_steps={cfg.ttt_steps}] success rate {summary['sr']:.3f} | "
          f"avg steps {summary['avg_steps']:.2f} | avg final angle {summary['avg_final_angle']:.3f}")
    print(f"Saved -> {os.path.join(out_dir, tag + '.pkl')}")
    return summary
