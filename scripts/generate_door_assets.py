#!/usr/bin/env python
"""Build the multimodal-door assets from PartNet-Mobility door objects.

    python scripts/generate_door_assets.py --pm-path ~/datasets/partnet-mobility \
        --output ~/datasets/failure_history_door

For every door object and movable link this writes up to four variants
``<obj>_<link>[_handle]_<mode>/`` that look identical when closed but open
differently:

    1  push right (original)      2  pull right (mirrored hinge direction)
    3  pull left  (hinge moved)   4  push left  (hinge moved + mirrored)

Variants with the handle removed get all four modes; variants that keep the
handle (``_handle``) get only push/pull, because a visible handle disambiguates
the hinge side.  All other joints of the object are frozen.

Ported from HAVE (``scripts/generate_assets/generate_multimodal_door.py``)
without the trimesh dependency; the split file ``metadata/multimodal_door.json``
lists the generated ids used by the paper.
"""

import argparse
import json
import math
import os
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from tqdm import tqdm

DOOR_IDS = ["8877", "8893", "8897", "8903", "8919", "8930", "8961", "8997", "9016", "9035", "9041",
            "9065", "9070", "9107", "9117", "9127", "9128", "9148", "9164", "9168", "9277", "9280",
            "9281", "9288", "9386", "9388", "9410", "8867", "8983", "8994", "9003", "9263", "9393"]
TWO_SIDED_DOORS = ["8877", "8919", "8930", "8961", "8997", "9016", "9128", "9168", "9280"]


def _obj_vertices(path):
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
    return np.asarray(verts, dtype=float)


def merged_link_vertices(link, urdf_dir):
    """All visual-mesh vertices of a link, concatenated."""
    chunks = []
    for visual in link.findall("visual"):
        mesh = visual.find("geometry/mesh") if visual.find("geometry") is not None else None
        if mesh is None:
            continue
        path = os.path.join(urdf_dir, mesh.get("filename"))
        if os.path.exists(path):
            chunks.append(_obj_vertices(path))
        else:
            print(f"warning: mesh {path} not found")
    return np.vstack(chunks) if chunks else np.zeros((0, 3))


def modify_urdf(input_dir, output_root, prefix, link_id, handle=True, axis_change=False, skip_first=False):
    tree = ET.parse(os.path.join(input_dir, "mobility.urdf"))
    root = tree.getroot()
    joint_id = link_id.replace("link", "joint")
    joint = root.find(f".//joint[@name='{joint_id}']")
    link = root.find(f".//link[@name='{link_id}']")
    if joint is None or link is None:
        raise ValueError(f"{joint_id}/{link_id} not found in {input_dir}")

    origin, axis, limit = joint.find("origin"), joint.find("axis"), joint.find("limit")
    lower, upper = float(limit.get("lower")), float(limit.get("upper"))
    joint_xyz = [float(x) for x in origin.get("xyz").split()]
    joint_axis = [float(x) for x in axis.get("xyz").split()]
    vertices = merged_link_vertices(link, input_dir)

    visual_origins = [v.find("origin") for v in link.findall("visual") if v.find("origin") is not None]
    collision_origins = [c.find("origin") for c in link.findall("collision") if c.find("origin") is not None]
    visual_xyz = [[float(x) for x in o.get("xyz").split()] for o in visual_origins]
    collision_xyz = [[float(x) for x in o.get("xyz").split()] for o in collision_origins]

    # Which side of the hinge the panel sits on, and its width (closed pose).
    angle = -joint_axis[1] * lower
    rot = np.array([[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]])
    rotated = (vertices + np.array(visual_xyz[0])[None, :]) @ rot.T
    min_x, max_x = rotated[:, 0].min(), rotated[:, 0].max()
    door_width = abs(max_x - min_x)
    default_is_right = abs(max_x) < abs(min_x)

    def move_link_origin(xyz):
        x, y, z = xyz
        x_diff = abs(door_width * math.cos(lower))
        new_x = (x + x_diff) if default_is_right else (x - x_diff)
        z_diff = abs(door_width * math.sin(lower))
        default_is_back = abs(-z - vertices[:, 2].min()) < abs(-z - vertices[:, 2].max())
        new_z = (z - z_diff) if default_is_back else (z + z_diff)
        return [new_x, y, new_z]

    def move_joint_origin(xyz):
        x, y, z = xyz
        return [x - door_width, y, z] if default_is_right else [x + door_width, y, z]

    mirrored_axis = [joint_axis[0], -joint_axis[1], joint_axis[2]]
    modes = {
        "1": dict(joint_xyz=joint_xyz, axis=joint_axis, visual=visual_xyz, collision=collision_xyz,
                  lower=lower, upper=upper),
        "2": dict(joint_xyz=joint_xyz, axis=mirrored_axis, visual=visual_xyz, collision=collision_xyz,
                  lower=-lower, upper=upper - 2 * lower),
        "3": dict(joint_xyz=move_joint_origin(joint_xyz), axis=joint_axis,
                  visual=[move_link_origin(v) for v in visual_xyz],
                  collision=[move_link_origin(c) for c in collision_xyz], lower=lower, upper=upper),
        "4": dict(joint_xyz=move_joint_origin(joint_xyz), axis=mirrored_axis,
                  visual=[move_link_origin(v) for v in visual_xyz],
                  collision=[move_link_origin(c) for c in collision_xyz],
                  lower=-lower, upper=upper - 2 * lower),
    }

    # Freeze every other joint at its lower limit (double doors).
    for other in root.findall("joint"):
        if other.get("name") == joint_id:
            other.set("type", "revolute")
            continue
        if other.get("type") == "revolute":
            o_limit = other.find("limit")
            o_lower = float(o_limit.get("lower"))
            o_limit.set("lower", "0")
            rot_dir = float(other.find("axis").get("xyz").split()[1])
            other.find("origin").set("rpy", f"0 {rot_dir * o_lower} 0")
        other.set("type", "fixed")

    for mode, v in modes.items():
        if skip_first and mode == "1":
            continue
        if (handle or not axis_change) and int(mode) >= 3:
            continue
        origin.set("xyz", " ".join(map(str, v["joint_xyz"])))
        axis.set("xyz", " ".join(map(str, v["axis"])))
        limit.set("lower", str(v["lower"]))
        limit.set("upper", str(v["upper"]))
        for o, xyz in zip(visual_origins, v["visual"]):
            o.set("xyz", " ".join(map(str, xyz)))
        for o, xyz in zip(collision_origins, v["collision"]):
            o.set("xyz", " ".join(map(str, xyz)))

        out_dir = os.path.join(output_root, f"{prefix}{'_handle' if handle else ''}_{mode}")
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        shutil.copytree(input_dir, out_dir)
        tree.write(os.path.join(out_dir, "mobility.urdf"))
        print(f"saved {out_dir}")


def create_no_handle_replicate(input_dir, keep_links, output_dir):
    """Copy the object with every movable part except ``keep_links`` removed.
    Returns True when something (a handle) was removed."""
    shutil.copytree(input_dir, output_dir, dirs_exist_ok=True)
    urdf_out = os.path.join(output_dir, "mobility.urdf")
    tree = ET.parse(os.path.join(input_dir, "mobility.urdf"))
    root = tree.getroot()

    joints_to_remove, links_to_remove = [], set()
    for joint in root.findall("joint"):
        child = joint.find("child").get("link")
        movable = joint.get("type") == "revolute" or (joint.get("type") != "fixed" and joint.find("axis") is not None)
        if movable and child not in keep_links:
            joints_to_remove.append(joint)
            links_to_remove.add(child)
    for joint in joints_to_remove:
        root.remove(joint)
    for link in root.findall("link"):
        if link.get("name") in links_to_remove:
            root.remove(link)

    sem_path = os.path.join(output_dir, "semantics.txt")
    with open(sem_path) as f:
        lines = f.readlines()
    with open(sem_path, "w") as f:
        f.writelines([ln for ln in lines if not any(r in ln for r in links_to_remove)])
    tree.write(urdf_out)
    return len(links_to_remove) != 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pm-path", required=True, help="PartNet-Mobility root containing raw/<id>/")
    parser.add_argument("--output", required=True, help="output root; assets go to <output>/raw/")
    parser.add_argument("--movable-links", default="metadata/movable_links_fullset_000_full.json")
    args = parser.parse_args()

    with open(args.movable_links) as f:
        movable_links = json.load(f)
    out_raw = os.path.join(os.path.expanduser(args.output), "raw")
    pm_raw = os.path.join(os.path.expanduser(args.pm_path), "raw")

    for obj in tqdm(DOOR_IDS):
        for link in movable_links[obj]:
            obj_id = f"{obj}_{link}"
            src = os.path.join(pm_raw, obj)
            no_handle = os.path.join(out_raw, f"{obj_id}_1")
            had_handle = create_no_handle_replicate(src, movable_links[obj], no_handle)
            if had_handle:
                modify_urdf(src, out_raw, obj_id, link, handle=True)
            # Without a handle only push/pull are ambiguous unless the door can
            # also be re-hinged (single doors).
            modify_urdf(no_handle, out_raw, obj_id, link, handle=False,
                        axis_change=obj not in TWO_SIDED_DOORS, skip_first=True)


if __name__ == "__main__":
    main()
