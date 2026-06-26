# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import numpy as np
import trimesh

from graspgenx.dataset.eval_utils import load_from_isaac_grasp_format
from graspgenx.utils.viser_utils import (
    create_visualizer,
    get_color_from_score,
    visualize_mesh,
    visualize_x_grasp,
)
from graspgenx.x_grippers import resolve_gripper_info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize saved GraspGenX Isaac-grasp YAML outputs in Viser."
    )
    parser.add_argument(
        "--mesh_file",
        type=str,
        required=True,
        help="Object mesh used for grasp generation.",
    )
    parser.add_argument(
        "--grasps_file",
        type=str,
        required=True,
        help="YAML file written by demo_object_mesh.py --output_file.",
    )
    parser.add_argument(
        "--gripper_name",
        type=str,
        required=True,
        help="Gripper name, e.g. unitree_g1.",
    )
    parser.add_argument(
        "--mesh_scale",
        type=float,
        default=1.0,
        help="Scale factor to apply to the object mesh.",
    )
    parser.add_argument(
        "--assets_dir",
        type=str,
        default=None,
        help="Optional assets directory containing x_grippers/ and proc_grippers/.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Viser port.",
    )
    parser.add_argument(
        "--plot_gripper_meshes",
        action="store_true",
        help="Render the full gripper collision mesh for visible grasps.",
    )
    parser.add_argument(
        "--max_gripper_meshes",
        type=int,
        default=10,
        help="Maximum full gripper meshes to render when --plot_gripper_meshes is set.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=-1,
        help="Only visualize the top K grasps by confidence. Default: all.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    mesh_file = Path(args.mesh_file)
    grasps_file = Path(args.grasps_file)
    if not mesh_file.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_file}")
    if not grasps_file.exists():
        raise FileNotFoundError(f"Grasps file not found: {grasps_file}")

    obj_mesh = trimesh.load(str(mesh_file), force="mesh")
    obj_mesh.apply_scale(args.mesh_scale)

    grasps, confidences = load_from_isaac_grasp_format(str(grasps_file))
    order = np.argsort(-confidences)
    if args.topk > 0:
        order = order[: args.topk]
    grasps = grasps[order]
    confidences = confidences[order]

    gripper = resolve_gripper_info(args.gripper_name, assets_dir=args.assets_dir)
    colors = get_color_from_score(confidences, use_255_scale=True)

    vis = create_visualizer(port=args.port)
    visualize_mesh(vis, "object_mesh", obj_mesh, color=[169, 169, 169])

    for i, (grasp, confidence) in enumerate(zip(grasps, confidences)):
        color = [0, 100, 255] if i == 0 else colors[i]
        linewidth = 5.0 if i == 0 else 3.0
        visualize_x_grasp(
            vis,
            f"saved_grasps/grasp_{i:03d}_score_{confidence:.3f}",
            grasp,
            color=color,
            gripper_info=gripper,
            linewidth=linewidth,
        )
        if args.plot_gripper_meshes and i < args.max_gripper_meshes:
            visualize_mesh(
                vis,
                f"saved_gripper_meshes/grasp_{i:03d}_score_{confidence:.3f}",
                gripper.collision_mesh,
                color=color,
                transform=grasp,
            )

    print(f"Loaded {len(grasps)} grasps from {grasps_file}")
    print(f"Top confidence: {confidences[0]:.3f}")
    print(f"Visualization ready: http://localhost:{args.port}")
    input("Press Enter to exit.")


if __name__ == "__main__":
    main()
