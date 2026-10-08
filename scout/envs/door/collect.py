"""Offline trajectory generation for the ambiguous-door benchmark.

Every door asset ``<id>_<link>_<mode>`` exists in several *modes* (hinge on a
different side, slider, ...) that look identical when closed.  A trajectory
is built as:

  1. H wrong interactions: the ground-truth flow of a *wrong* mode (rendered
     from that mode's own asset and articulated to the current joint angle)
     is executed with the suction gripper on the real door.  The observed
     per-point flow is recorded as the outcome.
  2. One correct interaction: the real mode's ground-truth flow, after which
     the door is advanced by 20 % of its range.
  3. A set of candidate query actions at the final state (ground truth,
     the articulated history actions, random actions) is executed and
     scored for the outer loop.

Trajectory keys: ``pcds, flows, 3dafs, obs_flows, masks, points, directions,
scores, evaluate{pcds, flows, 3dafs, points, directions, scores}, obj_id,
target_link, mode_id``.
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


def generate_trajectory(pm_dir, door_id, target_link, real_mode_id, has_handle=False,
                        history_len=-1, evaluate_random_action_cnt=10):
    mode_cnt = 2 if has_handle else 4

    # 1. Initialise every mode to get its initial GT flow (articulated later).
    all_initial_modes, main_env, main_gt_model, main_raw_data = [], None, None, None
    init_angle = target_angle = 0.0
    for m in range(mode_cnt):
        obj_id_m = f"{door_id}_{target_link}_{m + 1}"
        if not os.path.exists(os.path.join(pm_dir, obj_id_m)):
            mode_cnt = m
            break
        raw_data_m = PMObject(os.path.join(pm_dir, obj_id_m))
        env_m = _make_env(pm_dir, obj_id_m, target_link, raw_data_m, [-4, 0, 4], True)

        available_joints = [j.name for j in raw_data_m.semantics.by_type("hinge")
                            + raw_data_m.semantics.by_type("slider")]
        for joint in available_joints:
            info = p.getJointInfo(env_m.render_env.obj_id, env_m.render_env.link_name_to_index[joint],
                                  env_m.render_env.client_id)
            env_m.set_joint_state(joint, info[8])
            if joint == target_link and m == real_mode_id:
                init_angle, target_angle = info[8], info[9]

        pc_obs_m = env_m.render(filter_nonobj_pts=True, n_pts=N_PTS)
        gt_model_m = GTFlowModel(raw_data_m, env_m)
        all_initial_modes.append({
            "raw_data": raw_data_m, "pcd": pc_obs_m[4], "pc_seg": pc_obs_m[5],
            "gt_flow": gt_model_m(pc_obs_m),
            "link_name_to_index": env_m.render_env.link_name_to_index.copy(),
            "T_world_base": env_m.render_env.T_world_base,
            "available_joints": available_joints,
            "init_joint_angles": env_m.render_env.get_joint_angles(),
        })
        if m != real_mode_id:
            p.disconnect(env_m.render_env.client_id)
        else:
            main_env, main_raw_data, main_gt_model = env_m, raw_data_m, gt_model_m

    if main_env is None:
        return None, None

    traj = {k: [] for k in ("pcds", "flows", "3dafs", "obs_flows", "scores", "points", "directions", "masks")}
    inter = {"pc_segs": [], "joint_angles": [], "angles": []}

    # 2. Trial-and-error history: H wrong modes then the real one.
    wrong_modes = [m for m in range(mode_cnt) if m != real_mode_id]
    if history_len == -1:
        history_len = np.random.randint(1, 8)
    sampled_wrong = (np.random.choice(wrong_modes, size=history_len, replace=True).tolist()
                     if wrong_modes else [])
    interaction_modes = sampled_wrong + [real_mode_id]
    angles = []

    # 3. Execute the sequence.
    for i, mode in enumerate(interaction_modes):
        pc_obs = main_env.render(filter_nonobj_pts=True, n_pts=N_PTS)
        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = pc_obs
        link_ixs = pc_seg == main_env.render_env.link_name_to_index[target_link]
        if len(P_world[link_ixs]) == 0:
            p.disconnect(main_env.render_env.client_id)
            return None, None

        curr_angle = main_env.get_joint_value(target_link)
        inter["joint_angles"].append(main_env.render_env.get_joint_angles())
        inter["angles"].append(curr_angle)
        inter["pc_segs"].append(pc_seg)
        curr_rel_angle = (curr_angle - init_angle) / (target_angle - init_angle)
        angles.append(curr_rel_angle)
        if i != 0:
            traj["scores"].append((angles[-1] - angles[-2]) / 0.1)

        pred_flow = main_gt_model(pc_obs)

        if mode == real_mode_id:
            best_flow_ixs, _, _ = choose_grasp_points(pred_flow[link_ixs], P_world[link_ixs],
                                                      filter_edge=False, k=1)
            best_flow = pred_flow[link_ixs][best_flow_ixs[0]].copy()
            best_point = P_world[link_ixs][best_flow_ixs[0]]
        else:
            wd = all_initial_modes[mode]
            init_pcd, init_flow = wd["pcd"], wd["gt_flow"]
            art = lambda pts: articulate_joint(wd["raw_data"], wd["init_joint_angles"].copy(),
                                                   target_link, curr_angle - init_angle, pts,
                                                   wd["pc_seg"], wd["link_name_to_index"], wd["T_world_base"])
            new_pcd = art(init_pcd)
            wrong_flow = art(init_pcd + init_flow) - new_pcd
            wrong_ixs = wd["pc_seg"] == wd["link_name_to_index"][target_link]
            gi = np.argmax(np.linalg.norm(wrong_flow[wrong_ixs], axis=1))
            best_flow = wrong_flow[wrong_ixs][gi] / np.maximum(np.linalg.norm(wrong_flow[wrong_ixs][gi]), 1e-5)
            best_point = new_pcd[wrong_ixs][gi]

        grasp_field = _grasp_flow_field(P_world, best_point, best_flow, link_ixs, sharpness=5)
        traj["pcds"].append(P_world)
        traj["flows"].append(grasp_field)
        traj["3dafs"].append(pred_flow)
        traj["masks"].append(link_ixs)
        traj["points"].append(best_point)
        traj["directions"].append(best_flow)
        # Execute.
        if mode != real_mode_id:
            contact, repeat = False, 0
            while not contact and repeat < 50:
                _, contact = main_env.teleport(best_point[np.newaxis, :], best_flow[np.newaxis, :],
                                               target_link=target_link)
                repeat += 1
            if contact:
                main_env.attach()
                main_env.pull_with_constraint(best_flow, n_steps=100, target_link=target_link, constraint=False)
                main_env.reset_gripper(target_link)
                p.stepSimulation(main_env.render_env.client_id)
        else:   # correct action: advance the door by 20 % of its range
            main_env.set_joint_state(target_link, init_angle + (curr_rel_angle + 0.2) * (target_angle - init_angle))

        P_world_new = articulate_joint(
            main_raw_data, main_env.render_env.get_joint_angles(), target_link,
            main_env.get_joint_value(target_link) - curr_angle, P_world, pc_seg,
            main_env.render_env.link_name_to_index, main_env.render_env.T_world_base)
        traj["obs_flows"].append(P_world_new - P_world)

    final_rel_angle = (main_env.get_joint_value(target_link) - init_angle) / (target_angle - init_angle)
    angles.append(final_rel_angle)
    traj["scores"].append((angles[-1] - angles[-2]) / 0.1)
    for k in ("pcds", "flows", "3dafs", "obs_flows", "masks", "points", "directions"):
        traj[k] = np.stack(traj[k], axis=0)
    traj["scores"] = np.array(traj["scores"])
    inter["pc_segs"] = np.stack(inter["pc_segs"], axis=0)

    # 4. Query actions at the final state: GT, articulated history, random.
    start_angle = inter["angles"][-1]
    ev = {"pcds": [], "flows": [], "3dafs": [], "scores": [], "points": [], "directions": []}
    T = len(traj["pcds"])
    for t, (pcd, flow, mask, pc_seg, joint_angle, angle) in enumerate(zip(
            traj["pcds"], traj["flows"], traj["masks"], inter["pc_segs"], inter["joint_angles"], inter["angles"])):
        if t == T - 1:
            ev["pcds"].append(pcd)
            ev["flows"].append(flow)
            ev["3dafs"].append(pred_flow)
            gi = np.argmax(np.linalg.norm(flow, axis=1))
            ev["points"].append(pcd[gi])
            ev["directions"].append(flow[gi])
        else:
            art = lambda pts: articulate_joint(main_raw_data, joint_angle, target_link, start_angle - angle,
                                                   pts, pc_seg, main_env.render_env.link_name_to_index,
                                                   main_env.render_env.T_world_base)
            new_pcd = art(pcd)
            new_flow = art(pcd + flow) - new_pcd
            ev["pcds"].append(new_pcd)
            ev["flows"].append(new_flow)
            gi = np.argmax(np.linalg.norm(new_flow, axis=1))
            ev["points"].append(new_pcd[gi])
            ev["directions"].append(new_flow[gi])

    for _ in range(evaluate_random_action_cnt - len(ev["pcds"])):
        gt_mask, gt_pcd = traj["masks"][-1], traj["pcds"][-1]
        best_point = gt_pcd[gt_mask][np.random.choice(len(gt_pcd[gt_mask]))]
        best_flow = np.random.normal(size=3)
        best_flow /= np.maximum(np.linalg.norm(best_flow), 1e-5)
        ev["pcds"].append(gt_pcd)
        ev["flows"].append(_grasp_flow_field(gt_pcd, best_point, best_flow, gt_mask, sharpness=1))
        ev["points"].append(best_point)
        ev["directions"].append(best_flow)

    # Score every query action by executing it from the final state.
    for pcd, flow, point, direction in zip(ev["pcds"], ev["flows"], ev["points"], ev["directions"]):
        main_env.set_joint_state(target_link, start_angle)
        _, _, _, _, P_world, _, _ = main_env.render(filter_nonobj_pts=True, n_pts=N_PTS)
        order = np.argsort(np.sum((P_world - point) ** 2, axis=1))
        contact = False
        for idx in order[:50]:
            _, contact = main_env.teleport(P_world[idx][np.newaxis, :], direction[np.newaxis, :],
                                           target_link=target_link)
            if contact:
                break
        if not contact:
            score = 0
        else:
            main_env.attach()
            main_env.pull_with_constraint(direction, n_steps=100, target_link=target_link, constraint=False)
            main_env.reset_gripper(target_link)
            p.stepSimulation(main_env.render_env.client_id)
            score = ((main_env.get_joint_value(target_link) - start_angle) / (target_angle - init_angle)) / 0.1
        ev["scores"].append(score)
    traj["evaluate"] = ev

    p.disconnect(main_env.render_env.client_id)
    return traj, final_rel_angle


def parse_obj_id(obj: str):
    """``'8867_link_1_3'`` -> (door_id='8867', target_link='link_1', mode_id=2)."""
    parts = obj.split("_")
    return parts[0], "_".join(parts[1:3]), int(parts[-1]) - 1


def generate_trajectories_batch(obj_ids, pm_dir, output_path, traj_per_obj, max_trials,
                                evaluate_random_action_cnt, history_len):
    trajectories = []
    print(f"Generating for {len(obj_ids)} objects -> {output_path}")
    for obj in tqdm(obj_ids):
        door_id, target_link, mode_id = parse_obj_id(obj)
        n_ok, n_try = 0, 0
        while n_ok < traj_per_obj and n_try < max_trials:
            traj, _ = generate_trajectory(pm_dir, door_id, target_link, mode_id,
                                             has_handle="handle" in obj, history_len=history_len,
                                             evaluate_random_action_cnt=evaluate_random_action_cnt)
            if traj is not None and len(traj["scores"]) == len(traj["pcds"]):
                traj.update(obj_id=door_id, target_link=target_link, mode_id=mode_id)
                trajectories.append(traj)
                n_ok += 1
            n_try += 1
        with open(output_path, "wb") as f:
            pkl.dump(trajectories, f)
    with open(output_path, "wb") as f:
        pkl.dump(trajectories, f)
    print(f"Saved {len(trajectories)} trajectories to {output_path}")


def collect_door_dataset(pm_dir, metadata_path, output_dir, num_processes=16,
                         traj_per_obj_train=100, traj_per_obj_test=20,
                         max_trials_train=200, max_trials_test=50,
                         evaluate_random_action_cnt=10, history_len=-1,
                         train_key="train-train", test_key="test", max_objects=None):
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
                           evaluate_random_action_cnt, history_len)
                          for i, chunk in enumerate(chunks) if chunk])
    print(f"Raw door trajectories written to {raw_dir}. Run scripts/prepare_door_dataset.py next.")
