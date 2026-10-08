"""Floating suction gripper (vendored from ``rpad.pybullet_envs.suction_gripper``).

The gripper is two free-floating bodies, a visual-only base and a suction
head with a thin collision cylinder at the tip, joined by a compliant fixed
constraint.  ``activate`` welds the tip to whatever object link it touches.
URDF/mesh assets come from Google's ravens (Apache-2.0), see ``assets/suction``.
"""

import os

import numpy as np
import pybullet as p

_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "suction")
SUCTION_BASE_URDF = os.path.join(_ASSET_DIR, "suction-base.urdf")
SUCTION_HEAD_URDF = os.path.join(_ASSET_DIR, "suction-head.urdf")


class FloatingSuctionGripper:
    def __init__(self, client_id):
        self.client_id = client_id

        base_pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
        self.base_id = p.loadURDF(SUCTION_BASE_URDF, *base_pose, physicsClientId=client_id)
        tip_pose = ((0.487, 0.109, 0.347), p.getQuaternionFromEuler((np.pi, 0, 0)))
        self.body_id = p.loadURDF(SUCTION_HEAD_URDF, *tip_pose, physicsClientId=client_id)

        # A heavy gripper keeps the motion stable when dragging an object.
        self.mass = 10
        p.changeDynamics(bodyUniqueId=self.base_id, linkIndex=0, mass=self.mass / 2, physicsClientId=client_id)
        p.changeDynamics(bodyUniqueId=self.body_id, linkIndex=0, mass=self.mass / 2, physicsClientId=client_id)

        # Compliant link between base and head so contact can bounce a little.
        cid = p.createConstraint(parentBodyUniqueId=self.base_id, parentLinkIndex=-1,
                                 childBodyUniqueId=self.body_id, childLinkIndex=-1,
                                 jointType=p.JOINT_FIXED, jointAxis=(0, 0, 0),
                                 parentFramePosition=(0, 0, 0), childFramePosition=(0, 0, -0.08),
                                 physicsClientId=client_id)
        p.changeConstraint(cid, maxForce=10000, physicsClientId=client_id)

        self.activated = False
        self.contact_const = None
        self.contact_link_index = None

    def set_pose(self, pos, ori):
        p.resetBasePositionAndOrientation(self.body_id, pos, ori, self.client_id)
        p.resetBasePositionAndOrientation(self.base_id, pos, ori, self.client_id)

    def set_velocity(self, lin_vel, ang_vel):
        p.resetBaseVelocity(self.body_id, lin_vel, ang_vel, self.client_id)
        p.resetBaseVelocity(self.base_id, lin_vel, ang_vel, self.client_id)

    def detect_contact(self, obj_id) -> bool:
        points = p.getContactPoints(bodyA=self.body_id, bodyB=obj_id, linkIndexA=0,
                                    physicsClientId=self.client_id)
        return len(points) != 0

    def activate(self, obj_id):
        """Weld the tip to the first object link currently in contact."""
        if self.activated:
            return
        points = p.getContactPoints(bodyA=self.body_id, bodyB=obj_id, linkIndexA=0,
                                    physicsClientId=self.client_id)
        if not points:
            return
        point = points[0]
        contact_pos_on_A = point[5]
        obj_id, contact_link = point[2], point[4]

        tip_pos, tip_ori = p.getLinkState(self.body_id, 0, physicsClientId=self.client_id)[:2]
        T_tip_contact = p.multiplyTransforms(*p.invertTransform(tip_pos, tip_ori), contact_pos_on_A, tip_ori)

        obj_pos, obj_ori = p.getLinkState(obj_id, contact_link, physicsClientId=self.client_id)[:2]
        T_obj_contact = p.multiplyTransforms(*p.invertTransform(obj_pos, obj_ori), contact_pos_on_A, tip_ori)

        self.contact_const = p.createConstraint(
            parentBodyUniqueId=self.body_id, parentLinkIndex=0,
            childBodyUniqueId=obj_id, childLinkIndex=contact_link,
            jointType=p.JOINT_FIXED, jointAxis=(0, 0, 0),
            parentFramePosition=T_tip_contact[0], parentFrameOrientation=T_tip_contact[1],
            childFramePosition=T_obj_contact[0], childFrameOrientation=T_obj_contact[1],
            physicsClientId=self.client_id)
        self.activated = True
        self.contact_link_index = contact_link

    def release(self):
        if self.contact_const is not None:
            p.removeConstraint(self.contact_const, physicsClientId=self.client_id)
        self.contact_const = None
        self.activated = False
        self.contact_link_index = None
