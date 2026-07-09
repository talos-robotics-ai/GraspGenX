#!/usr/bin/env python3
"""Generate the cuRobo robot config for the Unitree G1 + Dex3 right arm.

The end2end pipeline drives a robot through a single cuRobo ``robot_cfg`` YAML
(URDF path, base link, tool frame, active joints, per-link collision spheres).
The Franka / UR10e demos reuse cuRobo's shipped configs or a generated one
(``build_ur10e_gripper.py``); the G1 has none, so this script writes one.

Design (see ``docs/g1_dex3_end2end.md``):

  * **base_link = pelvis** — the URDF root. The end2end pipeline FKs the *full*
    URDF (rooted at pelvis) for rendering and uses a single ``robot_base_T``
    for both cuRobo grasp-transforms and that FK, so cuRobo's kinematic root
    must equal the URDF root. cuRobo therefore models the whole body, but only
    the **7 right-arm joints are active** (``cspace.joint_names``); the legs,
    waist, left arm and both hands are **locked** at the AMO standing pose
    (``lock_joints``). In Stage A the base is fixed; in Stage B the AMO policy
    drives the (unlocked, real) legs+waist in Newton while cuRobo still plans
    the arm against this quasi-static standing model.
  * **tool_frame = right_hand_palm_link** — matches the GraspGenX ``unitree_g1``
    gripper grasp frame (palm, approach along +Z).
  * **collision spheres** are fitted per link from the link's collision mesh:
    an oriented bounding box, then a line of spheres along its longest axis.
    Coarse but sufficient for arm↔table / arm↔torso avoidance.

Output (``end2end/curobo_assets/g1_dex3.yml``) uses the pipeline's
``ABSOLUTE_PATH_PLACEHOLDER_*`` tokens for the URDF + asset root, which
``e2e_grasp_demo.resolve_curobo_robot_dict`` substitutes from the robot YAML
(``${SAGE}/assets/g1/...``) at load time — so the committed file is portable.

Run::

    uv run python end2end/build_g1_dex3.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
import yaml

from paths import curobo_assets_dir, sage_dir

_HERE = Path(__file__).resolve().parent
CUROBO_ASSETS = _HERE / "curobo_assets"

PLACEHOLDER_URDF = "ABSOLUTE_PATH_PLACEHOLDER_URDF"
PLACEHOLDER_ASSET_ROOT = "ABSOLUTE_PATH_PLACEHOLDER_ASSET_ROOT"

# ---------------------------------------------------------------------------
# Canonical G1 29-body joint order + AMO standing pose (from SAGE-Grasp:
# sage_grasp/sim/wholebody/joints.py G1_JOINT_NAME_MAP and control_stack.py
# AMOObservationBuilder.default_dof_pos). Kept here verbatim so this generator
# has no import dependency on the SAGE-Grasp package.
# ---------------------------------------------------------------------------
G1_BODY_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
AMO_STANDING = np.array([
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    0.0, 0.0, 0.0,
    0.5, 0.0, 0.2, 0.3, 0.0, 0.0, 0.0,
    0.5, 0.0, -0.2, 0.3, 0.0, 0.0, 0.0,
])
RIGHT_ARM_JOINTS = G1_BODY_JOINTS[22:29]
RIGHT_ARM_STANDING = AMO_STANDING[22:29]

# Dex3 right-hand joints (baked into g1_body29_hand14.urdf). Locked open (0)
# for arm planning — the grasp close is driven by the robot profile, not cuRobo.
RIGHT_HAND_JOINTS = [
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
]
LEFT_HAND_JOINTS = [j.replace("right_", "left_") for j in RIGHT_HAND_JOINTS]

# Links that get collision spheres. The active right-arm chain must be covered
# so cuRobo avoids the table/torso while planning; the torso column + a coarse
# left-arm stub keep the arm from sweeping through the body.
ARM_CHAIN = [
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link",
    "right_wrist_yaw_link", "right_hand_palm_link",
]
BODY_LINKS = ["pelvis", "torso_link"]
# NOTE: the left arm is intentionally *excluded* from the collision model. It is
# locked in the AMO standing pose (left_shoulder_pitch=0.5, i.e. raised forward),
# which sits inside the right arm's forward workspace; including its coarse
# spheres made cuRobo flag a self-collision for essentially every right-arm
# grasp config. A right-arm reach to a tabletop object in front-right never
# actually strikes the left arm, so dropping it removes the false positives.
COLLISION_LINKS = BODY_LINKS + ARM_CHAIN


def _geom_to_mesh(geom, origin, asset_root: Path):
    """Turn one URDF geometry (mesh OR cylinder/box/sphere primitive) into a
    trimesh, transformed by its ``origin``. Returns None if unresolvable."""
    T = np.eye(4) if origin is None else np.asarray(origin, float)
    if geom.mesh is not None:
        mp = asset_root / geom.mesh.filename
        if not mp.is_file():
            return None
        try:
            m = trimesh.load(mp, force="mesh")
        except Exception:
            return None
        if geom.mesh.scale is not None:
            m.apply_scale(geom.mesh.scale)
    elif geom.cylinder is not None:
        m = trimesh.creation.cylinder(radius=geom.cylinder.radius,
                                      height=geom.cylinder.length)
    elif geom.box is not None:
        m = trimesh.creation.box(extents=geom.box.size)
    elif geom.sphere is not None:
        m = trimesh.creation.icosphere(radius=geom.sphere.radius)
    else:
        return None
    m.apply_transform(T)
    return m


def _shipped_g1_spheres() -> dict:
    """Load cuRobo's shipped, validated G1 collision spheres, keyed by link.

    The fork ships ``content/configs/robot/unitree_g1.yml`` — a proper cuRobo
    config for this exact robot with dense, hand-tuned per-link spheres and the
    same link names as the SAGE URDF. Preferring these over our coarse OBB fits
    is what makes cuRobo stop reporting false self-collisions. Returns {} if the
    shipped config can't be found (then everything falls back to OBB).
    """
    cfg_path = curobo_assets_dir().parent / "configs/robot/unitree_g1.yml"
    if not cfg_path.is_file():
        return {}
    doc = yaml.safe_load(cfg_path.read_text())

    def _find(x):
        if isinstance(x, dict):
            if "collision_spheres" in x:
                return x["collision_spheres"]
            for v in x.values():
                r = _find(v)
                if r is not None:
                    return r
        return None

    return _find(doc) or {}


def _link_collision_mesh(urdf, link_name: str, asset_root: Path):
    """Concatenate a link's collision geometry into one trimesh in link frame.

    Handles primitive collisions (the shoulders use cylinders) and falls back
    to the link's *visual* meshes for links with no collision block (pelvis).
    """
    link = urdf.link_map[link_name]
    sources = list(link.collisions) or list(link.visuals)
    pieces = []
    for src in sources:
        m = _geom_to_mesh(src.geometry, src.origin, asset_root)
        if m is not None and len(m.vertices):
            pieces.append(m)
    if not pieces:
        return None
    return trimesh.util.concatenate(pieces) if len(pieces) > 1 else pieces[0]


def _fit_spheres(mesh: trimesh.Trimesh, spacing: float = 0.06,
                 min_r: float = 0.02, max_r: float = 0.09) -> list[dict]:
    """Fit a line of spheres to a link mesh via its oriented bounding box.

    Lay spheres along the OBB's longest axis; radius ~ half the mean of the
    two shorter extents (clamped). Small links collapse to a single sphere.
    """
    try:
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    except Exception:
        # Degenerate mesh — one sphere at the centroid.
        c = mesh.bounds.mean(axis=0)
        r = float(np.clip(np.linalg.norm(mesh.extents) / 2, min_r, max_r))
        return [{"center": [round(float(x), 5) for x in c], "radius": round(r, 5)}]

    obb_to_link = np.linalg.inv(to_origin)
    axis = int(np.argmax(extents))
    length = float(extents[axis])
    others = [extents[i] for i in range(3) if i != axis]
    radius = float(np.clip(0.5 * float(np.mean(others)), min_r, max_r))

    n = max(1, int(np.ceil(length / spacing)) + 1)
    ts = np.linspace(-length / 2, length / 2, n) if n > 1 else [0.0]
    spheres = []
    for t in ts:
        c_obb = np.zeros(3)
        c_obb[axis] = t
        c_link = (obb_to_link @ np.array([*c_obb, 1.0]))[:3]
        spheres.append({
            "center": [round(float(x), 5) for x in c_link],
            "radius": round(radius, 5),
        })
    return spheres


def _chain_joints(urdf, base_link: str, target_link: str) -> list[str]:
    """Actuated joint names on the URDF path base_link -> target_link.

    cuRobo prunes joints that don't lead to a tool frame or collision link, so
    ``lock_joints`` may only reference joints on these retained chains — locking
    a pruned joint (e.g. a leg) raises KeyError in the kinematics loader.
    """
    child_to_joint = {j.child: j for j in urdf.robot.joints}
    chain, link = [], target_link
    while link != base_link:
        j = child_to_joint.get(link)
        if j is None:
            break  # reached a root other than base_link
        if j.type in ("revolute", "prismatic", "continuous"):
            chain.append(j.name)
        link = j.parent
    return chain


def _self_collision_ignore(present: list[str], collision_spheres: dict,
                           urdf, stand_cfg: dict) -> dict:
    """Auto-generate the self-collision ignore list from the resting pose.

    This mirrors how cuRobo's shipped robot configs are built: FK the standing
    configuration, then ignore any link pair whose spheres already overlap
    there — those are structurally-adjacent links (e.g. the short wrist links
    nest together) that can never meaningfully collide, and leaving them checked
    flags a permanent self-collision so cuRobo marks *every* IK solution
    infeasible. Coarse OBB spheres make several 2-apart pairs overlap, so a
    fixed "immediate neighbours only" list is not enough — we detect overlaps
    geometrically instead. A small positive margin also ignores pairs that sit
    just shy of touching, for robustness across arm configurations.
    """
    import numpy as _np
    urdf.update_cfg(stand_cfg)
    world = {}
    for link in present:
        T = urdf.get_transform(frame_to=link, frame_from="pelvis")
        centers = [(T @ _np.array([*s["center"], 1.0]))[:3] for s in collision_spheres[link]]
        radii = [s["radius"] for s in collision_spheres[link]]
        world[link] = (centers, radii)

    MARGIN = 0.02  # ignore pairs whose spheres are within 2 cm at rest
    ignore = {l: set() for l in present}
    for i in range(len(present)):
        for k in range(i + 1, len(present)):
            a, b = present[i], present[k]
            ca, ra = world[a]
            cb, rb = world[b]
            min_sep = min(
                float(_np.linalg.norm(pa - pb)) - (r1 + r2)
                for pa, r1 in zip(ca, ra)
                for pb, r2 in zip(cb, rb)
            )
            if min_sep < MARGIN:
                ignore[a].add(b)
                ignore[b].add(a)

    # Also ignore the arm against its own base body (torso + pelvis). Those
    # links carry coarse OBB spheres that falsely collide with the arm as it
    # sweeps near the trunk (the FK overlap check only covers the *standing*
    # pose, not the whole motion). The real arm can't strike its own base for a
    # tabletop reach in front of the robot, so this removes false positives
    # while keeping arm↔arm and arm↔left-stub checks intact.
    base_body = {"torso_link", "pelvis"} & set(present)
    arm_links = [l for l in ARM_CHAIN if l in present]
    for base in base_body:
        for arm in arm_links:
            ignore[base].add(arm)
            ignore[arm].add(base)
    return {l: sorted(v) for l, v in ignore.items() if v}


def build(output: Path) -> Path:
    import yourdfpy

    sage = sage_dir()
    urdf_path = sage / "assets/g1/g1_body29_hand14.urdf"
    asset_root = sage / "assets/g1"
    urdf = yourdfpy.URDF.load(str(urdf_path), build_collision_scene_graph=False,
                              load_meshes=False)

    print(f"[build_g1_dex3] URDF: {urdf_path}")
    shipped = _shipped_g1_spheres()
    if shipped:
        print(f"[build_g1_dex3] using shipped cuRobo G1 spheres ({len(shipped)} links)")
    collision_spheres = {}
    for link in COLLISION_LINKS:
        if link in shipped:
            collision_spheres[link] = shipped[link]
            print(f"  {link:28} {len(shipped[link]):2d} spheres (shipped)")
            continue
        mesh = _link_collision_mesh(urdf, link, asset_root)
        if mesh is None:
            print(f"  ! no collision mesh for {link}; skipping")
            continue
        spheres = _fit_spheres(mesh)
        collision_spheres[link] = spheres
        print(f"  {link:28} {len(spheres):2d} spheres (OBB fallback)")

    present = [l for l in COLLISION_LINKS if l in collision_spheres]

    # cuRobo retains only joints on the chains base_link -> {tool_frame,
    # collision links}. lock_joints must be a subset of those (minus the active
    # right-arm joints) — typically the waist (path to the right arm) plus the
    # left-arm joints leading to the left collision stub. Legs / left wrist /
    # hands are pruned, so they must NOT appear here.
    stand = {n: float(v) for n, v in zip(G1_BODY_JOINTS, AMO_STANDING)}
    retained: set[str] = set()
    for link in ["right_hand_palm_link", *present]:
        retained.update(_chain_joints(urdf, "pelvis", link))
    lock = {
        name: round(stand.get(name, 0.0), 4)
        for name in sorted(retained)
        if name not in RIGHT_ARM_JOINTS
    }

    robot_cfg = {
        "robot_cfg": {
            "kinematics": {
                "urdf_path": PLACEHOLDER_URDF,
                "asset_root_path": PLACEHOLDER_ASSET_ROOT,
                "base_link": "pelvis",
                "collision_sphere_buffer": 0.0,
                "use_global_cumul": True,
                "collision_link_names": present,
                "collision_spheres": collision_spheres,
                "cspace": {
                    "joint_names": list(RIGHT_ARM_JOINTS),
                    "cspace_distance_weight": [1.0] * 7,
                    "null_space_weight": [1.0] * 7,
                    "max_acceleration": 12.0,
                    "max_jerk": 500.0,
                    "position_limit_clip": 0.0,
                    "default_joint_position": [round(float(v), 4)
                                               for v in RIGHT_ARM_STANDING],
                },
                # A dedicated grasp/TCP frame. The pipeline assumes the tool
                # frame's +Z is the approach/lift axis (grasp_approach_axis="z"),
                # but the Dex3 palm approaches along its +X (fingers extend +X in
                # right_hand_palm_link). So attach a fixed frame rotated +90°
                # about the palm's Y, making the TCP's +Z coincide with the
                # palm's +X — i.e. the real finger/approach direction. cuRobo
                # then drives *this* frame to the GraspGenX grasp pose, so both
                # the grasp orientation and the approach/lift waypoints are
                # correct (grasp_to_tool_transform can stay identity).
                "extra_links": {
                    "right_dex3_tcp": {
                        "parent_link_name": "right_hand_palm_link",
                        "link_name": "right_dex3_tcp",
                        "joint_name": "right_dex3_tcp_joint",
                        "joint_type": "FIXED",
                        # [x, y, z, qw, qx, qy, qz]; quat = R_y(+90°).
                        "fixed_transform": [0.0, 0.0, 0.0,
                                            0.70710678, 0.0, 0.70710678, 0.0],
                    }
                },
                "tool_frames": ["right_dex3_tcp"],
                "lock_joints": lock,
                "mesh_link_names": present,
                "self_collision_buffer": {l: 0.0 for l in present},
                "self_collision_ignore": _self_collision_ignore(
                    present, collision_spheres, urdf, stand),
            }
        }
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        yaml.safe_dump(robot_cfg, f, sort_keys=False, default_flow_style=None)
    print(f"[build_g1_dex3] wrote {output} "
          f"({len(present)} collision links, {len(lock)} locked joints)")
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=CUROBO_ASSETS / "g1_dex3.yml")
    args = ap.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
