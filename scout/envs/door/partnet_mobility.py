"""Minimal PartNet-Mobility object description (URDF tree + joint semantics).

Vendored and trimmed from ``rpad.partnet_mobility_utils`` (data.py / urdf.py).
An object directory holds ``mobility.urdf``, ``semantics.txt`` and, optionally,
``meta.json``.  Many PartNet-Mobility URDFs violate the URDF schema, which is
why the tree is parsed by hand instead of with a URDF library.
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from xml.etree import ElementTree as ET

import numpy as np

JOINT_TYPES = {"slider", "free", "hinge", "heavy", "static", "slider+"}


# ── semantics.txt ────────────────────────────────────────────────────────────
@dataclass
class JointSemantic:
    name: str      # link name as used in the URDF
    type: str      # one of JOINT_TYPES
    label: str     # semantic label (door, drawer, ...)


@dataclass
class Semantics:
    sems: List[JointSemantic]

    def by_name(self, name: str) -> JointSemantic:
        return {s.name: s for s in self.sems}[name]

    def by_type(self, joint_type: Union[str, Sequence[str]]) -> List[JointSemantic]:
        types = {joint_type} if isinstance(joint_type, str) else set(joint_type)
        return [s for s in self.sems if s.type in types]

    def by_label(self, label: str) -> List[JointSemantic]:
        return [s for s in self.sems if s.label == label]

    @staticmethod
    def from_file(fn: Union[str, Path]) -> "Semantics":
        with Path(fn).open("r") as f:
            lines = [ln for ln in f.read().split("\n") if ln.strip()]
        sems = []
        for line in lines:
            name, jt, label = line.split(" ")
            if jt not in JOINT_TYPES:
                raise ValueError(f"bad joint type '{jt}' in {fn}")
            sems.append(JointSemantic(name, jt, label))
        assert len({s.name for s in sems}) == len(sems), "duplicate link in semantics.txt"
        return Semantics(sems)


# ── mobility.urdf ────────────────────────────────────────────────────────────
@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: Optional[Tuple[np.ndarray, np.ndarray]] = None   # (xyz, rpy)
    axis: Optional[np.ndarray] = None
    limit: Optional[Tuple[float, float]] = None


@dataclass
class Link:
    name: str
    mesh_names: Set[str]


class PMTree:
    """The parsed URDF kinematic tree.  PartNet-Mobility roots are always ``base``."""

    ROOT_LINK = "base"

    def __init__(self, links: List[Link], joints: List[Joint]):
        self._links = links
        self._joints = joints
        self._linkmap = {l.name: i for i, l in enumerate(links)}
        self._childmap = {j.child: i for i, j in enumerate(joints)}
        self._jointmap = {j.name: i for i, j in enumerate(joints)}

    @property
    def links(self) -> List[Link]:
        return copy.deepcopy(self._links)

    @property
    def joints(self) -> List[Joint]:
        return copy.deepcopy(self._joints)

    def get_chain(self, link_name: str) -> List[Joint]:
        """Joints from the base down to (and including) the joint whose child is ``link_name``."""
        parent = {j.child: (j.parent, j) for j in self._joints}
        parent[self.ROOT_LINK] = (None, None)
        chain: List[Joint] = []
        name, joint = parent[link_name]
        while name is not None and joint is not None:
            chain = [copy.deepcopy(joint)] + chain
            name, joint = parent[name]
        return chain

    def get_joint(self, name: str) -> Joint:
        if name not in self._jointmap:
            raise ValueError(f"invalid joint name: {name}")
        return copy.deepcopy(self._joints[self._jointmap[name]])

    def get_joint_by_child(self, child: str) -> Joint:
        if child not in self._childmap:
            raise ValueError(f"invalid child link: {child}")
        return copy.deepcopy(self._joints[self._childmap[child]])

    def get_link(self, name: str) -> Link:
        if name not in self._linkmap:
            raise ValueError(f"invalid link name: {name}")
        return copy.deepcopy(self._links[self._linkmap[name]])

    @property
    def children(self) -> Dict[str, List[str]]:
        children: Dict[str, List[str]] = defaultdict(list)
        for link in self._links:
            children[link.name] = []
        for joint in self._joints:
            children[joint.parent].append(joint.child)
        return dict(children)

    @property
    def descendants(self) -> Dict[str, List[str]]:
        desc: Dict[str, List[str]] = defaultdict(list)
        children = self.children

        def dfs(name, ancestors):
            for a in ancestors:
                desc[a].append(name)
            desc[name] = []
            for c in children.get(name, []):
                dfs(c, ancestors + [name])

        dfs(self.ROOT_LINK, [])
        return dict(desc)

    @staticmethod
    def parse_urdf_from_string(urdf: str) -> "PMTree":
        robot = ET.fromstring(urdf)

        def pose(el):
            xyz = np.asarray(el.attrib["xyz"].split(), dtype=float) if "xyz" in el.attrib else np.zeros(3)
            rpy = np.asarray(el.attrib["rpy"].split(), dtype=float) if "rpy" in el.attrib else np.zeros(3)
            return xyz, rpy

        def link(el):
            return Link(el.attrib["name"], {it.attrib["filename"] for it in el.iter() if it.tag == "mesh"})

        def joint(el):
            origin_el, axis_el, limit_el = el.find("origin"), el.find("axis"), el.find("limit")
            axis = None
            if axis_el is not None:   # malformed entries use "None" for a component
                axis = np.asarray([s if s != "None" else "0" for s in axis_el.attrib["xyz"].split()], dtype=float)
            return Joint(
                name=el.attrib["name"], type=el.attrib["type"],
                parent=el.find("parent").attrib["link"], child=el.find("child").attrib["link"],
                origin=pose(origin_el) if origin_el is not None else None, axis=axis,
                limit=(float(limit_el.attrib["lower"]), float(limit_el.attrib["upper"]))
                if limit_el is not None else None,
            )

        return PMTree([link(e) for e in robot.findall("link")], [joint(e) for e in robot.findall("joint")])

    @staticmethod
    def parse_urdf(fn: Union[str, Path]) -> "PMTree":
        path = Path(fn)
        if not (path.exists() and path.suffix == ".urdf"):
            raise ValueError(f"{path} is not a URDF file")
        return PMTree.parse_urdf_from_string(path.read_text())

    def __repr__(self):
        return f"PMTree(links={len(self._links)}, joints={len(self._joints)})"


# ── object directory ─────────────────────────────────────────────────────────
class PMObject:
    """One PartNet-Mobility object directory."""

    def __init__(self, obj_dir: Union[str, Path]):
        self.obj_dir = Path(obj_dir)
        self.semantics = Semantics.from_file(self.obj_dir / "semantics.txt")
        self.obj = PMTree.parse_urdf(self.obj_dir / "mobility.urdf")
        meta = self.obj_dir / "meta.json"
        self.category = json.loads(meta.read_text()).get("model_cat", "") if meta.exists() else ""

    @property
    def obj_id(self) -> str:
        return self.obj_dir.name

    @property
    def urdf_fn(self) -> Path:
        return self.obj_dir / "mobility.urdf"

    def movable_links(self) -> List[str]:
        return [s.name for s in self.semantics.by_type("hinge") + self.semantics.by_type("slider")]

    def __repr__(self):
        return f'PMObject(id="{self.obj_id}")'
