# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sample-directory loaders + format auto-detection used by both demo scripts.
#
# Two input formats are supported:
#   - "realworld":  <NN>/{depth.npy, rgb.png, seg.png, meta_data.json}
#                   (back-projects depth via intrinsics + camera_pose,
#                   segments into obj_* per label_map)
#   - "json":       *.json files in GraspGenX format (object_info / scene_info)

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def depth_to_camera_xyz(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(H,W) depth (meters) + (3,3) K → (H,W,3) XYZ in camera frame."""
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth.astype(np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def transform_xyz(xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (..., 3) points."""
    return xyz @ T[:3, :3].T + T[:3, 3]


def _load_realworld_metadata(sample_dir: str) -> dict:
    """Read meta_data.json (intrinsics, camera_pose, label_map, scene_bounds)."""
    json_path = os.path.join(sample_dir, "meta_data.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No meta_data.json in {sample_dir}")
    with open(json_path, "r") as f:
        return json.load(f)


def load_realworld_scene(sample_dir: str, min_obj_points: int = 100) -> dict:
    """Load an M2T2-style real_world/<NN>/ scene directory.

    Returns a dict with:
        name, scene_xyz (Ns,3) world frame, scene_rgb (Ns,3) uint8,
        objects: {obj_<id>: {"pc","rgb","label_id"}},
        camera_pose, intrinsics, scene_bounds,
        _depth, _seg_2d, _scene_seg, _format (private fields used by
        ``build_scene_pc_excluding_object`` to mask out the target object
        without re-projecting depth).
    """
    md = _load_realworld_metadata(sample_dir)
    K = np.asarray(md["intrinsics"], dtype=np.float64)
    cam_pose = np.asarray(md["camera_pose"], dtype=np.float64)
    label_map = md["label_map"]
    sb = md.get("scene_bounds")
    # Labels listed here are excluded from .objects — typically because the
    # 2D segmentation for that obj_* is too noisy/wrong to produce a usable
    # point cloud. The companion "obj_denylist_reason" field carries the
    # human-readable cause (commonly "bad segmentations").
    obj_denylist = set(md.get("obj_denylist", []) or [])
    obj_denylist_reason = md.get("obj_denylist_reason", "bad segmentations")

    depth = np.load(os.path.join(sample_dir, "depth.npy")).astype(np.float32)
    rgb_uint8 = np.asarray(Image.open(os.path.join(sample_dir, "rgb.png")))[..., :3]
    seg_arr = np.asarray(
        Image.open(os.path.join(sample_dir, "seg.png")), dtype=np.int32
    )

    xyz_cam = depth_to_camera_xyz(depth, K)
    valid = depth > 0
    xyz_cam_flat = xyz_cam.reshape(-1, 3)
    rgb_flat = rgb_uint8.reshape(-1, 3)
    seg_flat = seg_arr.reshape(-1)
    valid_flat = valid.reshape(-1)

    scene_xyz_world = transform_xyz(xyz_cam_flat[valid_flat], cam_pose).astype(
        np.float32
    )
    scene_rgb = rgb_flat[valid_flat].astype(np.uint8)
    scene_seg = seg_flat[valid_flat].astype(np.int32)

    objects: Dict[str, dict] = {}
    skipped_denied: List[str] = []
    for name, lid in label_map.items():
        if not name.startswith("obj_"):
            continue
        if name in obj_denylist:
            skipped_denied.append(name)
            continue
        m = (seg_flat == int(lid)) & valid_flat
        if int(m.sum()) < min_obj_points:
            continue
        obj_xyz_world = transform_xyz(xyz_cam_flat[m], cam_pose).astype(np.float32)
        obj_rgb = rgb_flat[m].astype(np.uint8)
        objects[name] = {"pc": obj_xyz_world, "rgb": obj_rgb, "label_id": int(lid)}
    if skipped_denied:
        scene_name = os.path.basename(os.path.normpath(sample_dir))
        print(
            f"[scene_loaders] {scene_name}: skipped {skipped_denied} "
            f"({obj_denylist_reason})"
        )

    return {
        "name": os.path.basename(os.path.normpath(sample_dir)),
        "scene_xyz": scene_xyz_world,
        "scene_rgb": scene_rgb,
        "objects": objects,
        "camera_pose": cam_pose,
        "intrinsics": K,
        "scene_bounds": sb,
        "_depth": depth,
        "_seg_2d": seg_arr,
        "_scene_seg": scene_seg,
        "_format": "realworld",
    }


def load_graspgenx_json_scene(json_path: str) -> dict:
    """Load a GraspGenX JSON scene file: object_info + scene_info."""
    with open(json_path, "r") as f:
        data = json.load(f)

    obj_pc = np.array(data["object_info"]["pc"])
    obj_pc_color = np.array(data["object_info"]["pc_color"])

    full_pc_key = "pc_color" if "pc_color" in data["scene_info"] else "full_pc"
    xyz_scene = np.array(data["scene_info"][full_pc_key])[0]
    xyz_scene_color = np.array(data["scene_info"]["img_color"]).reshape(1, -1, 3)[
        0, :, :
    ]

    if "obj_mask" in data["scene_info"]:
        xyz_seg = np.array(data["scene_info"]["obj_mask"]).reshape(-1)
        xyz_scene = xyz_scene[xyz_seg != 1]
        xyz_scene_color = xyz_scene_color[xyz_seg != 1]

    return {
        "name": Path(json_path).stem,
        "scene_xyz": xyz_scene,
        "scene_rgb": xyz_scene_color,
        "objects": {"obj_0": {"pc": obj_pc, "rgb": obj_pc_color, "label_id": 0}},
        "_format": "json",
    }


def _load_ply_points_colors(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a point-cloud PLY -> (xyz (N,3) float32, rgb (N,3) uint8)."""
    import trimesh

    loaded = trimesh.load(path, process=False)
    xyz = np.asarray(loaded.vertices, dtype=np.float32)
    rgb = None
    colors = getattr(loaded, "colors", None)
    if colors is not None and len(colors) == len(xyz):
        rgb = np.asarray(colors, dtype=np.uint8)[:, :3]
    if rgb is None:
        visual = getattr(loaded, "visual", None)
        vc = getattr(visual, "vertex_colors", None) if visual is not None else None
        if vc is not None and len(vc) == len(xyz):
            rgb = np.asarray(vc, dtype=np.uint8)[:, :3]
    if rgb is None:
        rgb = np.full((len(xyz), 3), 160, dtype=np.uint8)
    return xyz, rgb


def _points_near_voxel(
    scene_xyz: np.ndarray, obj_xyz: np.ndarray, radius: float
) -> np.ndarray:
    """Boolean mask over ``scene_xyz`` of points within ~``radius`` m of the
    object cloud. Voxel-hash approximation (numpy-only, no scipy): a scene
    point is 'near' if its voxel lies in the 26-neighborhood of any occupied
    object voxel, i.e. within ~sqrt(3)*radius."""
    if len(obj_xyz) == 0 or len(scene_xyz) == 0:
        return np.zeros(len(scene_xyz), dtype=bool)
    occ = set(map(tuple, np.floor(obj_xyz / radius).astype(np.int64)))
    expanded = set()
    for (x, y, z) in occ:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    expanded.add((x + dx, y + dy, z + dz))
    svox = np.floor(scene_xyz / radius).astype(np.int64)
    return np.fromiter(
        (tuple(v) in expanded for v in svox), dtype=bool, count=len(svox)
    )


def load_r2r2r_scene(manifest_path: str) -> dict:
    """Load an R2R2R scene from a ``scene_pc.json`` manifest.

    The manifest (paths relative to its own directory) looks like::

        {
          "format": "r2r2r_scene_pc",
          "scene_pc": "scene_point_cloud.npz",   # keys: points (N,3), colors (N,3)
          "objects": {"white_cube": "obj0.ply", "cardboard_box": "obj1.ply"},
          "exclude_radius_m": 0.015
        }

    All clouds must be in the SAME metric (meters) world frame — for the
    Polycam pipeline that is the Nerfstudio frame (``--world-frame nerfstudio``),
    which matches the per-object ``point_cloud.ply`` exports.

    A per-point scene segmentation (``_scene_seg``) is derived by proximity so
    that :func:`build_scene_pc_excluding_object` can drop the target object's
    own points before collision checking (otherwise every grasp on an object
    would 'collide' with that object's surface in the scene cloud).
    """
    manifest_path = os.path.abspath(manifest_path)
    base = os.path.dirname(manifest_path)
    with open(manifest_path, "r") as f:
        mani = json.load(f)

    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(base, p)

    scene_data = np.load(_resolve(mani["scene_pc"]))
    scene_xyz = np.asarray(scene_data["points"], dtype=np.float32)
    if "colors" in scene_data:
        scene_rgb = np.asarray(scene_data["colors"], dtype=np.uint8)[:, :3]
    else:
        scene_rgb = np.full((len(scene_xyz), 3), 160, dtype=np.uint8)

    radius = float(mani.get("exclude_radius_m", 0.015))
    scene_seg = np.full(len(scene_xyz), -1, dtype=np.int32)
    objects: Dict[str, dict] = {}
    for lid, (label, obj_path) in enumerate(mani["objects"].items()):
        obj_xyz, obj_rgb = _load_ply_points_colors(_resolve(obj_path))
        objects[label] = {"pc": obj_xyz, "rgb": obj_rgb, "label_id": int(lid)}
        # Assign scene points near this object to its id (first-match wins for
        # the rare overlap case; the objects in R2R2R scenes are separated).
        near = _points_near_voxel(scene_xyz, obj_xyz, radius) & (scene_seg == -1)
        scene_seg[near] = int(lid)

    return {
        "name": mani.get("name", Path(base).name),
        "scene_xyz": scene_xyz,
        "scene_rgb": scene_rgb,
        "objects": objects,
        "_scene_seg": scene_seg,
        "_format": "r2r2r",
    }


def build_scene_pc_excluding_object(scene: dict, label: str) -> np.ndarray:
    """World-frame scene PC with the named object's pixels removed.

    For real_world and r2r2r scenes this just indexes the cached ``_scene_seg``
    aligned to ``scene_xyz`` — no depth re-projection. For JSON scenes the
    loader already excluded the (single) target, so we return ``scene_xyz``
    as-is.
    """
    # r2r2r (and any future format) that ships a per-point seg aligned to
    # scene_xyz: drop the target object's points by label id.
    if scene.get("_format") == "r2r2r":
        scene_xyz = np.asarray(scene["scene_xyz"], dtype=np.float32)
        obj = scene["objects"].get(label)
        scene_seg = scene.get("_scene_seg")
        if obj is None or scene_seg is None:
            return scene_xyz
        return scene_xyz[np.asarray(scene_seg) != int(obj["label_id"])]

    if scene.get("_format") != "realworld":
        return np.asarray(scene["scene_xyz"], dtype=np.float32)

    obj = scene["objects"].get(label)
    if obj is None:
        return np.asarray(scene["scene_xyz"], dtype=np.float32)
    target_id = int(obj["label_id"])

    scene_xyz = np.asarray(scene["scene_xyz"], dtype=np.float32)
    scene_seg = scene.get("_scene_seg")
    if scene_seg is None:
        # Older scene dicts without the cached per-point seg labels: fall
        # back to depth reprojection.
        depth = scene["_depth"]
        seg_2d = scene["_seg_2d"]
        K = scene["intrinsics"]
        cam_pose = scene["camera_pose"]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        mask = (depth > 0) & (seg_2d != target_id) & np.isfinite(depth)
        v, u = np.where(mask)
        if len(v) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        d = depth[v, u]
        x = (u - cx) * d / fx
        y = (v - cy) * d / fy
        cam_pts = np.stack([x, y, d], axis=1).astype(np.float32)
        return transform_xyz(cam_pts, cam_pose).astype(np.float32)

    keep = scene_seg != target_id
    return scene_xyz[keep]


def _find_r2r2r_manifest(sample_data_dir: str) -> str | None:
    """Return the path to an R2R2R scene manifest in this dir, or None.

    Recognized as either a file literally named ``scene_pc.json`` or any
    ``*.json`` whose top-level ``format`` is ``"r2r2r_scene_pc"``."""
    named = os.path.join(sample_data_dir, "scene_pc.json")
    if os.path.isfile(named):
        return named
    for f in sorted(glob.glob(os.path.join(sample_data_dir, "*.json"))):
        try:
            with open(f, "r") as fh:
                if json.load(fh).get("format") == "r2r2r_scene_pc":
                    return f
        except (json.JSONDecodeError, OSError):
            continue
    return None


def detect_format(sample_data_dir: str) -> str:
    """Return 'r2r2r' if the dir contains an R2R2R scene manifest, 'realworld'
    if it contains <NN>/meta_data.json subdirs, 'json' if it contains *.json
    files, else raise."""
    if _find_r2r2r_manifest(sample_data_dir) is not None:
        return "r2r2r"
    realworld_dirs = sorted(
        [
            d
            for d in glob.glob(os.path.join(sample_data_dir, "*"))
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "meta_data.json"))
        ]
    )
    json_files = sorted(glob.glob(os.path.join(sample_data_dir, "*.json")))
    if realworld_dirs:
        return "realworld"
    if json_files:
        return "json"
    raise FileNotFoundError(
        f"{sample_data_dir} contains neither <NN>/meta_data.json nor *.json files"
    )


def collect_scene_items(
    sample_data_dir: str, scene_filter: str | None = None
) -> List[Tuple[str, str]]:
    """Return [(fmt_tag, path)] for scene-level iteration (used by demo_scene_pc).

    For 'realworld', each item is a scene directory; for 'json', each item is
    one .json file. Optional ``scene_filter`` restricts to a single <NN> in
    real-world mode.
    """
    fmt = detect_format(sample_data_dir)
    if fmt == "r2r2r":
        return [("r2r2r", _find_r2r2r_manifest(sample_data_dir))]
    if fmt == "realworld":
        all_dirs = sorted(
            [
                d
                for d in glob.glob(os.path.join(sample_data_dir, "*"))
                if os.path.isdir(d)
                and os.path.exists(os.path.join(d, "meta_data.json"))
            ]
        )
        if scene_filter is not None:
            name = scene_filter.lstrip("/")
            all_dirs = [d for d in all_dirs if os.path.basename(d) == name]
            if not all_dirs:
                raise FileNotFoundError(
                    f"--scene {scene_filter} not found under {sample_data_dir}"
                )
        return [("realworld", d) for d in all_dirs]
    return [
        ("json", f) for f in sorted(glob.glob(os.path.join(sample_data_dir, "*.json")))
    ]


def collect_object_items(sample_data_dir: str, min_obj_points: int = 100) -> List[Dict]:
    """Flat list of per-object samples (used by demo_object_pc).

    Real-world format: one item per (scene, obj_*) pair.
    JSON format: one item per *.json file with a top-level "pc" field.

    Returns dicts with keys: name, pc, pc_color, stored_grasps, stored_grasp_conf.
    """
    fmt = detect_format(sample_data_dir)
    if fmt == "realworld":
        items: List[Dict] = []
        for d in sorted(glob.glob(os.path.join(sample_data_dir, "*"))):
            if not (
                os.path.isdir(d) and os.path.exists(os.path.join(d, "meta_data.json"))
            ):
                continue
            scene = load_realworld_scene(d, min_obj_points=min_obj_points)
            for label, obj in scene["objects"].items():
                items.append(
                    {
                        "name": f"{scene['name']}/{label}",
                        "pc": np.asarray(obj["pc"], dtype=np.float32),
                        "pc_color": np.asarray(obj["rgb"], dtype=np.uint8),
                        "stored_grasps": None,
                        "stored_grasp_conf": None,
                    }
                )
        return items

    json_files = sorted(glob.glob(os.path.join(sample_data_dir, "*.json")))
    items = []
    for f in json_files:
        with open(f, "r") as fh:
            data = json.load(fh)
        if "pc" not in data:
            continue
        has_stored = "grasp_poses" in data and "grasp_conf" in data
        items.append(
            {
                "name": Path(f).stem,
                "pc": np.array(data["pc"]),
                "pc_color": np.array(data["pc_color"]),
                "stored_grasps": np.array(data["grasp_poses"]) if has_stored else None,
                "stored_grasp_conf": (
                    np.array(data["grasp_conf"]) if has_stored else None
                ),
            }
        )
    return items
