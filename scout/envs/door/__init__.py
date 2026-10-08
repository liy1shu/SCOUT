"""Ambiguous articulated doors: PartNet-Mobility door assets simulated in PyBullet
with a floating suction gripper.

Self-contained.  The simulation stack is vendored (and trimmed) from the
HAVE / FlowBot3D / rpad code bases so no external robotics packages are needed:

partnet_mobility.py   URDF tree + semantics of one object (rpad.partnet_mobility_utils)
articulate.py         analytic articulation and ground-truth flow (rpad + flowbot3d)
render.py             PyBullet renderer -> segmented world-frame point cloud (rpad)
suction_gripper.py    floating suction gripper (rpad.pybullet_envs; ravens assets)
suction_sim.py        PMSuctionSim / GTFlowModel / choose_grasp_points (HAVE)
collect.py            offline trajectory generation
rollout.py            closed-loop evaluation

Assets: generate them from PartNet-Mobility with ``scripts/generate_door_assets.py``;
the split used in the paper is ``metadata/multimodal_door.json``.
"""
