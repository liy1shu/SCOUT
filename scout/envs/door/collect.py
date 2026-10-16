"""Offline trajectory generation for the ambiguous-door benchmark.

Follows HAVE_CoRL25 ``scripts/collect_dataset.py --door`` (the generator of the
training data of the released checkpoint).  Each trajectory runs
on the real door asset from the closed state for 1-30 steps:

  1. Every history step is the ground-truth action with prob. 0.2 while the door
     is closed (0.8 once it is open, >10 %), otherwise a perturbed-GT or random
     action that is actually executed (it may open or close the door).  The GT
     action advances the door by 10 % of its range.  The last step is always GT.
  2. Query actions at the current state (articulated history + GT, 3 perturbed
     GT, random actions) are executed and scored for the outer loop.

Trajectory keys: ``pcds, flows, 3dafs, obs_flows, masks, points, directions,
scores, evaluate{pcds, flows, 3dafs, points, directions, scores}, obj_id, joint_id``.
"""

from __future__ import annotations

import json
import os
import pickle as pkl
from multiprocessing import Pool
import numpy as np
import pybullet as p
from tqdm import tqdm

from scout.envs.door.articulate import articulate_joint
from scout.envs.door.partnet_mobility import PMObject
from scout.envs.door.suction_sim import GTFlowModel, PMSuctionSim, choose_grasp_points

N_PTS = 1200


def _grasp_flow_field(P_world, best_point, best_flow, link_ixs, sharpness):
    field = np.zeros_like(P_world) + best_flow[np.newaxis, :]
    coeff = np.exp(-sharpness * np.power(P_world - best_point, 2).sum(axis=-1))[:, np.newaxis]
    return field * coeff * link_ixs[:, np.newaxis]


def _make_env(pm_dir, obj_id, target_link, raw_data, camera_pos, camera_pos_random):
    env = PMSuctionSim(obj_id, pm_dir, gui=False, camera_pos=camera_pos,
                       camera_pos_random=camera_pos_random, gripper=True)
    env.disable_self_collision()
    for link in [joint.name for joint in raw_data.semantics.sems]:
        if link != target_link:
            env.disable_collision(env.render_env.link_name_to_index[link])
        else:
            env.disable_collision(env.render_env.link_name_to_index[link], body=False, floor=True)
    env.reset()
    return env


CLOSED_GT_RATE = 0.2   # P(GT action) while the door is still closed (rel. angle < 0.1)
OPEN_GT_RATE = 0.8     # P(GT action) once the door is open


def _normalize_flow(flow):
    return flow / (np.linalg.norm(flow, axis=-1).max() + 1e-6)


def generate_trajectory(pm_dir, obj_id, target_link, history_len=-1, evaluate_random_action_cnt=10):
    """One trajectory, following HAVE_CoRL25 ``scripts/collect_dataset.py``.

    ``obj_id`` is the full asset id (e.g. ``8867_link_1_3``); ``history_len`` is the step count (history + current
    step), -1 = 50 % uniform in [1, 30], 50 % uniform in [1, 9].
    """
    if history_len == -1:
        history_len = np.random.randint(1, 31) if np.random.randint(0, 2) == 1 else np.random.randint(1, 10)
    step_cnt = history_len

    raw_data = PMObject(os.path.join(pm_dir, obj_id))
    env = _make_env(pm_dir, obj_id, target_link, raw_data, [-4, 0, 4], True)
    gt_model = GTFlowModel(raw_data, env)

    # Close all joints.
    for link in [j.name for j in raw_data.semantics.by_type("hinge") + raw_data.semantics.by_type("slider")]:
        info = p.getJointInfo(env.render_env.obj_id, env.render_env.link_name_to_index[link], env.render_env.client_id)
        env.set_joint_state(link, info[8])
    info = p.getJointInfo(env.render_env.obj_id, env.render_env.link_name_to_index[target_link],
                          env.render_env.client_id)
    init_angle, target_angle = info[8], info[9]

    traj = {k: [] for k in ("pcds", "flows", "3dafs", "obs_flows", "scores", "points", "directions", "masks")}
    inter = {"pc_segs": [], "joint_angles": [], "angles": []}
    angles = []

    for i in range(step_cnt):
        pc_obs = env.render(filter_nonobj_pts=True, n_pts=N_PTS)
        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = pc_obs
        link_ixs = pc_seg == env.render_env.link_name_to_index[target_link]
        if len(P_world[link_ixs]) == 0:
            p.disconnect(env.render_env.client_id)
            return None, None

        curr_angle = env.get_joint_value(target_link)
        inter["joint_angles"].append(env.render_env.get_joint_angles())
        inter["angles"].append(curr_angle)
        inter["pc_segs"].append(pc_seg)
        curr_rel_angle = (curr_angle - init_angle) / (target_angle - init_angle)
        angles.append(curr_rel_angle)
        if i != 0:
            traj["scores"].append((angles[-1] - angles[-2]) / 0.1)

        pred_flow = gt_model(pc_obs)
        if curr_rel_angle > 0.98:   # already fully open: HAVE discards these trajectories
            p.disconnect(env.render_env.client_id)
            return None, None

        gt_rate = CLOSED_GT_RATE if curr_rel_angle < 0.1 else OPEN_GT_RATE
        randomize = (i != step_cnt - 1) and np.random.randint(100) > int(100 * gt_rate)

        if randomize:
            if np.random.randint(0, 2) == 0:
                # Perturbed GT: 0 = same direction + min-flow point, 1 = opposite + max point, 2 = opposite + min point
                mode = np.random.randint(0, 3)
                norms = np.linalg.norm(pred_flow[link_ixs], axis=1)
                best_flow = pred_flow[link_ixs][np.argmax(norms)].copy()
                if mode == 0:
                    grasp_ix = np.argmin(norms)
                elif mode == 1:
                    grasp_ix = np.argmax(norms)
                    best_flow = -best_flow
                else:
                    grasp_ix = np.argmin(norms)
                    best_flow = -best_flow
                best_point = P_world[link_ixs][grasp_ix]
                best_flow /= np.maximum(np.linalg.norm(best_flow), 1e-5)
            else:   # fully random
                best_point = P_world[link_ixs][np.random.choice(len(P_world[link_ixs]))]
                best_flow = np.random.normal(size=3)
                best_flow /= np.maximum(np.linalg.norm(best_flow), 1e-5)

            contact, repeat = False, 0
            while not contact and repeat < 50:
                _, contact = env.teleport(best_point[np.newaxis, :], best_flow[np.newaxis, :], target_link=target_link)
                repeat += 1
            if not contact:
                p.disconnect(env.render_env.client_id)
                return None, None
        else:
            best_flow_ixs, _, _ = choose_grasp_points(pred_flow[link_ixs], P_world[link_ixs], filter_edge=False, k=1)
            best_flow = pred_flow[link_ixs][best_flow_ixs[0]].copy()
            best_point = P_world[link_ixs][best_flow_ixs[0]]

        grasp_field = _grasp_flow_field(P_world, best_point, best_flow, link_ixs, sharpness=5)
        traj["pcds"].append(P_world)
        traj["flows"].append(grasp_field)
        traj["3dafs"].append(pred_flow)
        traj["masks"].append(link_ixs)
        traj["points"].append(best_point)
        traj["directions"].append(best_flow)

        if randomize:   # execute the (wrong / random) action on the real door
            env.attach()
            env.pull_with_constraint(best_flow, n_steps=100, target_link=target_link, constraint=False)
            env.reset_gripper(target_link)
            p.stepSimulation(env.render_env.client_id)
        else:           # GT action: advance the door by 10 % of its range
            env.set_joint_state(target_link, init_angle + (curr_rel_angle + 0.1) * (target_angle - init_angle))

        P_world_new = articulate_joint(
            raw_data, env.render_env.get_joint_angles(), target_link,
            env.get_joint_value(target_link) - curr_angle, P_world, pc_seg,
            env.render_env.link_name_to_index, env.render_env.T_world_base)
        traj["obs_flows"].append(P_world_new - P_world)

    final_rel_angle = (env.get_joint_value(target_link) - init_angle) / (target_angle - init_angle)
    angles.append(final_rel_angle)
    traj["scores"].append((angles[-1] - angles[-2]) / 0.1)
    for k in ("pcds", "flows", "3dafs", "obs_flows", "masks", "points", "directions"):
        traj[k] = np.stack(traj[k], axis=0)
    traj["scores"] = np.array(traj["scores"])
    inter["pc_segs"] = np.stack(inter["pc_segs"], axis=0)

    # Query actions at the current state: articulated history + GT, 3 perturbed GT, random.
    start_angle = inter["angles"][-1]
    ev = {"pcds": [], "flows": [], "3dafs": [], "scores": [], "points": [], "directions": []}
    T = len(traj["pcds"])
    for t, (pcd, flow, pc_seg, joint_angle, angle) in enumerate(zip(
            traj["pcds"], traj["flows"], inter["pc_segs"], inter["joint_angles"], inter["angles"])):
        if t == T - 1:
            ev["pcds"].append(pcd)
            ev["flows"].append(flow)
            ev["3dafs"].append(pred_flow)
            gi = np.argmax(np.linalg.norm(flow, axis=1))
            ev["points"].append(pcd[gi])
            ev["directions"].append(flow[gi])
        else:
            art = lambda pts: articulate_joint(raw_data, joint_angle, target_link, start_angle - angle, pts, pc_seg,
                                               env.render_env.link_name_to_index, env.render_env.T_world_base)
            new_pcd = art(pcd)
            new_flow = art(pcd + flow) - new_pcd
            ev["pcds"].append(new_pcd)
            ev["flows"].append(new_flow)
            gi = np.argmax(np.linalg.norm(new_flow, axis=1))
            ev["points"].append(new_pcd[gi])
            ev["directions"].append(new_flow[gi])

    gt_mask, gt_pcd = traj["masks"][-1], traj["pcds"][-1]
    for mode in range(3):
        art_flow = _normalize_flow(pred_flow)
        art_norms = np.linalg.norm(art_flow, axis=1)[:, np.newaxis]
        gt_flow_mask = (art_norms >= 0.1).squeeze()
        art_norms = art_norms[gt_flow_mask]
        norms = np.linalg.norm(traj["flows"][-1][gt_mask], axis=1)
        best_flow = pred_flow[gt_mask][np.argmax(norms)].copy()
        if mode == 0:
            grasp_ix = np.argmin(norms)
            art_flow[gt_flow_mask] = (1 - art_norms) * art_flow[gt_flow_mask] / art_norms
        elif mode == 1:
            grasp_ix = np.argmax(norms)
            best_flow = -best_flow
            art_flow[gt_flow_mask] = -1 * art_flow[gt_flow_mask]
        else:
            grasp_ix = np.argmin(norms)
            best_flow = -best_flow
            art_flow[gt_flow_mask] = (art_norms - 1) * art_flow[gt_flow_mask] / art_norms
        best_point = gt_pcd[gt_mask][grasp_ix]
        best_flow /= np.maximum(np.linalg.norm(best_flow), 1e-5)
        ev["pcds"].append(gt_pcd)
        ev["flows"].append(_grasp_flow_field(gt_pcd, best_point, best_flow, gt_mask, sharpness=1))
        ev["3dafs"].append(art_flow)
        ev["points"].append(best_point)
        ev["directions"].append(best_flow)

    for _ in range(evaluate_random_action_cnt - 3):
        best_point = gt_pcd[gt_mask][np.random.choice(len(gt_pcd[gt_mask]))]
        best_flow = np.random.normal(size=3)
        best_flow /= np.maximum(np.linalg.norm(best_flow), 1e-5)
        ev["pcds"].append(gt_pcd)
        ev["flows"].append(_grasp_flow_field(gt_pcd, best_point, best_flow, gt_mask, sharpness=1))
        ev["points"].append(best_point)
        ev["directions"].append(best_flow)

    # Score every query action by executing it from the current state.
    for pcd, flow, point, direction in zip(ev["pcds"], ev["flows"], ev["points"], ev["directions"]):
        env.set_joint_state(target_link, start_angle)
        _, _, _, _, P_world, _, _ = env.render(filter_nonobj_pts=True, n_pts=N_PTS)
        order = np.argsort(np.sum((P_world - point) ** 2, axis=1))
        contact = False
        for idx in order[:50]:
            _, contact = env.teleport(P_world[idx][np.newaxis, :], direction[np.newaxis, :], target_link=target_link)
            if contact:
                break
        if not contact:
            score = 0
        else:
            env.attach()
            env.pull_with_constraint(direction, n_steps=100, target_link=target_link, constraint=False)
            env.reset_gripper(target_link)
            p.stepSimulation(env.render_env.client_id)
            score = ((env.get_joint_value(target_link) - start_angle) / (target_angle - init_angle)) / 0.1
        ev["scores"].append(score)
    traj["evaluate"] = ev

    p.disconnect(env.render_env.client_id)
    return traj, final_rel_angle


def parse_obj_id(obj: str):
    """``'8867_link_1_3'`` -> (door_id='8867', target_link='link_1', mode_id=2)."""
    parts = obj.split("_")
    return parts[0], "_".join(parts[1:3]), int(parts[-1]) - 1


def generate_trajectories_batch(obj_ids, pm_dir, output_path, traj_per_obj, max_trials,
                                evaluate_random_action_cnt, history_len, seed=None):
    if seed is not None:
        np.random.seed(seed)
    trajectories = []
    print(f"Generating for {len(obj_ids)} objects -> {output_path}")
    for obj in tqdm(obj_ids):
        _, target_link, _ = parse_obj_id(obj)
        n_ok, n_try = 0, 0
        while n_ok < traj_per_obj and n_try < max_trials:
            traj, _ = generate_trajectory(pm_dir, obj, target_link, history_len=history_len,
                                          evaluate_random_action_cnt=evaluate_random_action_cnt)
            if traj is not None and len(traj["scores"]) == len(traj["pcds"]):
                traj.update(obj_id=obj, joint_id=target_link)
                trajectories.append(traj)
                n_ok += 1
            n_try += 1
        with open(output_path, "wb") as f:
            pkl.dump(trajectories, f)
    with open(output_path, "wb") as f:
        pkl.dump(trajectories, f)
    print(f"Saved {len(trajectories)} trajectories to {output_path}")


def collect_door_dataset(pm_dir, metadata_path, output_dir, num_processes=16,
                         traj_per_obj_train=200, traj_per_obj_test=0,
                         max_trials_train=400, max_trials_test=100,
                         evaluate_random_action_cnt=10, history_len=-1,
                         train_key="train-train", test_key="test", max_objects=None, seed=42):
    """Collect raw pickles into ``<output_dir>/raw_pkls/{train,test}_<worker>.pkl``."""
    raw_dir = os.path.join(output_dir, "raw_pkls")
    os.makedirs(raw_dir, exist_ok=True)
    with open(metadata_path) as f:
        split = json.load(f)

    for key, tag, per_obj, max_trials in ((train_key, "train", traj_per_obj_train, max_trials_train),
                                          (test_key, "test", traj_per_obj_test, max_trials_test)):
        obj_ids = split.get(key, [])[:max_objects]
        chunks = [obj_ids[i::num_processes] for i in range(num_processes)]
        with Pool(num_processes) as pool:
            pool.starmap(generate_trajectories_batch,
                         [(chunk, pm_dir, os.path.join(raw_dir, f"{tag}_{i}.pkl"), per_obj, max_trials,
                           evaluate_random_action_cnt, history_len, seed + i + (0 if tag == "train" else 1000))
                          for i, chunk in enumerate(chunks) if chunk])
    print(f"Raw door trajectories written to {raw_dir}. Run scripts/prepare_door_dataset.py next.")
