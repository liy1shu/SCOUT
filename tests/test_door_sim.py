"""End-to-end smoke test of the self-contained door simulation (needs pybullet + scipy).

Builds a synthetic two-mode door in PartNet-Mobility layout (a hinged panel
that opens by pulling in mode 1 and by pushing in mode 2), then runs:

  * analytic articulation / ground-truth flow sanity checks
  * offline trajectory generation (collect.generate_trajectory)
  * a short closed-loop rollout with an untrained SCOUT model

Run with ``python tests/test_door_sim.py`` (a few seconds on CPU).
"""

import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

_URDF = """<?xml version="1.0"?>
<robot name="door">
  <link name="base">
    <visual><origin xyz="0 -0.55 1.0"/><geometry><box size="0.1 0.1 2.0"/></geometry></visual>
    <collision><origin xyz="0 -0.55 1.0"/><geometry><box size="0.1 0.1 2.0"/></geometry></collision>
  </link>
  <link name="link_0">
    <visual><origin xyz="0 0.5 0"/><geometry><box size="0.05 1.0 1.9"/></geometry>
      <material name="wood"><color rgba="0.6 0.4 0.2 1"/></material></visual>
    <collision><origin xyz="0 0.5 0"/><geometry><box size="0.05 1.0 1.9"/></geometry></collision>
    <inertial><mass value="1.0"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="joint_0" type="revolute">
    <origin xyz="0 -0.5 1.0" rpy="0 0 0"/>
    <axis xyz="0 0 {axis_z}"/>
    <parent link="base"/>
    <child link="link_0"/>
    <limit lower="0" upper="1.4" effort="100" velocity="1"/>
  </joint>
</robot>
"""


def make_synthetic_doors(root):
    """Writes ``<root>/door_link_0_1`` (opens toward -x) and ``..._2`` (opens toward +x)."""
    for mode, axis_z in ((1, 1), (2, -1)):
        d = os.path.join(root, f"door_link_0_{mode}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "mobility.urdf"), "w") as f:
            f.write(_URDF.format(axis_z=axis_z))
        with open(os.path.join(d, "semantics.txt"), "w") as f:
            f.write("link_0 hinge door\n")
        with open(os.path.join(d, "meta.json"), "w") as f:
            f.write('{"model_cat": "Door"}')


def test_articulation():
    from scout.envs.door.articulate import articulate_points, fk
    from scout.envs.door.partnet_mobility import Joint

    j = Joint("j", "revolute", "base", "l", origin=(np.array([1.0, 0, 0]), np.zeros(3)), axis=np.array([0, 0, 1.0]))
    T = fk([j], [np.pi / 2])
    assert np.allclose(T[:3, 3], [1, 0, 0]) and np.allclose(T[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-9)
    moved = articulate_points(np.array([[2.0, 0, 0]]), np.eye(4), [j], [0.0], [np.pi / 2])
    assert np.allclose(moved, [[1.0, 1.0, 0]], atol=1e-9), moved
    print("  articulation: ok")


def test_collect_and_rollout():
    import torch
    from scout.configs import get_task_config
    from scout.envs.door import collect, rollout
    from scout.envs.door.partnet_mobility import PMObject
    from scout.envs.door.suction_sim import GTFlowModel, PMSuctionSim
    from scout.models.scout import SCOUT

    root = tempfile.mkdtemp()
    try:
        make_synthetic_doors(root)
        obj = PMObject(os.path.join(root, "door_link_0_1"))
        assert obj.movable_links() == ["link_0"]

        # Render + ground-truth flow: the panel should move toward -x in mode 1.
        env = PMSuctionSim("door_link_0_1", root, camera_pos=[-3, 0, 2])
        env.disable_self_collision()
        obs = env.render(filter_nonobj_pts=True, n_pts=collect.N_PTS)
        P_world, pc_seg = obs[4], obs[5]
        assert len(P_world) == collect.N_PTS
        link_ixs = pc_seg == env.render_env.link_name_to_index["link_0"]
        assert link_ixs.sum() > 100, "door panel not visible"
        flow = GTFlowModel(obj, env)(obs)
        assert flow[link_ixs][:, 0].mean() < 0, "mode-1 panel should move toward the camera (-x)"
        assert (~link_ixs).sum() == 0 or np.abs(flow[~link_ixs]).max() == 0, "static links must have zero flow"
        assert np.isclose(np.linalg.norm(flow, axis=-1).max(), 1.0, atol=1e-4)
        print("  render + gt flow: ok")

        # The GT flow is state dependent: it rotates with the door, and it is the
        # same whether the chain is evaluated at zero angles (as HAVE does) or at
        # the true angles, because the hinge axis is fixed in the world frame.
        from scout.envs.door.articulate import compute_normalized_flow
        closed_dir = flow[link_ixs].mean(0)
        env.set_joint_state("link_0", 0.8)
        obs_open = env.render(filter_nonobj_pts=True, n_pts=collect.N_PTS)
        m_open = obs_open[5] == env.render_env.link_name_to_index["link_0"]
        flow_open = GTFlowModel(obj, env)(obs_open)
        flow_true = compute_normalized_flow(obs_open[4], env.render_env.T_world_base,
                                            env.render_env.get_joint_angles(), obs_open[5],
                                            env.render_env.link_name_to_index, obj)
        assert np.abs(flow_open - flow_true).max() < 1e-9
        open_dir = flow_open[m_open].mean(0)
        cos = np.dot(closed_dir, open_dir) / (np.linalg.norm(closed_dir) * np.linalg.norm(open_dir))
        assert cos < 0.9, "flow direction must change with the door state"
        env.set_joint_state("link_0", 0.0)
        print("  gt flow at open state: ok")

        # Physics: approach the strongest-flow point with the suction gripper and pull.
        from scout.envs.door.suction_sim import choose_grasp_points
        env.disable_collision(env.render_env.link_name_to_index["link_0"], body=False, floor=True)
        _, flows, points = choose_grasp_points(flow[link_ixs], P_world[link_ixs], k=5)
        ix, contact = env.teleport(points, flows, target_link="link_0")
        assert contact, "suction tip should reach the door panel"
        env.attach()
        env.pull_with_constraint(flows[ix], n_steps=100, target_link="link_0", constraint=True)
        opened = env.get_joint_value("link_0")
        assert opened > 0.05, f"pulling along the gt flow should open the door (angle {opened:.3f})"
        env.close()
        print(f"  suction pull: ok (joint angle {opened:.2f} rad after one pull)")

        np.random.seed(0)
        traj, final_rel = collect.generate_trajectory(root, "door", "link_0", real_mode_id=0,
                                                      history_len=1, evaluate_random_action_cnt=3)
        assert traj is not None
        T = len(traj["pcds"])
        assert T == 2 and traj["obs_flows"].shape == (T, collect.N_PTS, 3)
        assert traj["points"].shape == (T, 3) and traj["directions"].shape == (T, 3)
        assert len(traj["scores"]) == T and len(traj["evaluate"]["scores"]) == 3
        assert np.linalg.norm(traj["obs_flows"][-1][traj["masks"][-1]], axis=-1).max() > 0.01, \
            "correct action should move the door"
        print(f"  generate_trajectory: ok (final rel angle {final_rel:.2f})")

        cfg = get_task_config("door").override(**{"model.d_model": 32, "model.nhead": 2, "model.gmm_layers": 1,
                                                  "model.dyn_layers": 1, "model.latent_dim": 16,
                                                  "rollout.max_attempts": 2, "rollout.ttt_steps": 1})
        torch.manual_seed(0)
        model = SCOUT(cfg.model).eval()
        res = rollout.rollout_door(model, cfg.rollout, root, "door", "link_0", mode_id=0)
        assert res is not None
        step_id, success, angle, logs = res
        assert step_id in (0, 1) and isinstance(success, bool) and "gmm_weights" in logs
        print(f"  rollout_door: ok (steps {step_id + 1}, success {success}, angle {angle:.2f})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_articulation()
    test_collect_and_rollout()
    print("door sim tests passed")
