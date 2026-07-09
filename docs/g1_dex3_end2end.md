# Unitree G1 + Dex3 in the end2end pipeline

This document describes how the **Unitree G1 humanoid + Dex3 right hand** is
integrated into the GraspGenX `end2end` grasping pipeline, mirroring the Franka
Panda demo's structure. It is the reference for the two-stage plan, the files
added, the joint/frame conventions, and how to run it.

> **Control split (the design goal).** The **upper body** (right-arm IK +
> collision-free motion planning) is owned by **cuRobo**. The **lower body**
> (legs + waist balance) is owned by the **AMO policy** from SAGE-Grasp,
> running inside the Newton physics replay. The Dex3 right hand is the gripper.

---

## 1. Status

| Stage | Scope | State |
|---|---|---|
| **A** | Mirror the Franka structure: robot/env YAML + profile + cuRobo config; G1 loads, GraspGenX predicts Dex3 grasps, cuRobo plans the right arm, **kinematic** playback + MP4/USD export. Fixed base. | **Structure complete; cuRobo grasp-plan not yet landing** (§9) |
| **B** | Floating base in Newton + AMO lower-body controller in the `dynamic_playback` substep loop; the G1 balances while the arm executes the pick. | **Planned** (design in §7) |

Stage A changes nothing about how a robot is selected or imported — it is a new
`--robot_config` YAML exactly like `franka_panda.yaml`, and **no existing
pipeline file was modified** (only `paths.py` gained a `${SAGE}` token and
`robot_profiles.py` gained the `G1Dex3Profile`).

**What is verified working (§9 has the detail):**
- The G1 + Dex3 loads; the robot/env/profile wiring resolves end-to-end
  (URDF FK over all 43 joints, `${SAGE}` expansion, cuRobo placeholder
  substitution).
- **GraspGenX predicts real Dex3 grasps** (gripper `unitree_g1`, 80 grasps,
  confidence 0.97–0.99).
- The cuRobo config is **valid and self-consistent**: base=`pelvis`, the 7
  right-arm joints active, the rest locked at standing, a synthetic **TCP frame**
  fixing the Dex3 approach axis, and an **FK-derived self-collision ignore list**
  so the model is no longer permanently self-colliding.

**What is not yet landing:** cuRobo's `plan_grasp` does not yet return a
collision-free plan for the tabletop scene — the standing G1's right arm reaching
top-down grasps near its reach limit, next to the table, with approach/lift
waypoints, is geometrically tight. This is scene/collision-model tuning, not a
structural gap (see §9).

---

## 2. What SAGE-Grasp already provides

Nothing about the robot or the policy is re-implemented; it is re-hosted:

- `SAGE-Grasp/assets/g1/g1_body29_hand14.urdf` — the **full** G1: 29 body DOF +
  **14 Dex3 hand DOF baked in** (7 per hand: `thumb_0/1/2`, `index_0/1`,
  `middle_0/1`). No separate hand attachment is needed.
- `SAGE-Grasp/checkpoints/{amo_jit.pt, adapter_jit.pt, adapter_norm_stats.pt}` —
  the AMO lower-body policy + adapter (used in Stage B).
- `SAGE-Grasp/sage_grasp/sim/wholebody/control_stack.py` — the MuJoCo-facing
  AMO observation builder + policy runner. Stage B ports its observation
  contract onto Newton state (see §7).
- GraspGenX gripper **`unitree_g1`** (`ext/gripper_descriptions/.../x_grippers/
  unitree_g1`) — a `revolute_3f` model of the Dex3 hand with a palm-centred
  grasp frame (approach along +Z). This is what GraspGenX predicts grasps for.

---

## 3. Files added / changed

All under `GraspGenX/end2end/` unless noted. **No existing task or import path
was modified** — the additions slot into the existing dispatch tables.

| File | Change | Purpose |
|---|---|---|
| `paths.py` | +`${SAGE}` token (`sage_dir()`) | Resolve the sibling SAGE-Grasp checkout for G1 assets/checkpoints (override `$GRASPGENX_SAGE_DIR`). |
| `robot_profiles.py` | +`G1Dex3Profile` (registered as `g1_dex3`) | Right-arm joints, Dex3 gripper joints + open/close, tool frame, AMO lower-body fields. |
| `build_g1_dex3.py` | **new** generator | Emits the cuRobo config (collision spheres fitted from the URDF meshes). Idempotent; re-run any time. |
| `curobo_assets/g1_dex3.yml` | **new** (generated) | cuRobo `robot_cfg`: base=`pelvis`, ee=`right_hand_palm_link`, 7 active right-arm joints, everything else locked at standing. |
| `robots/g1_dex3.yaml` | **new** | The `--robot_config` entry, mirroring `franka_panda.yaml`. |
| `envs/g1_tabletop_demo.yaml` | **new** | A `pick_and_lift` tabletop scene tuned for the standing G1's right-arm reach. |
| `../docs/g1_dex3_end2end.md` | **new** | This document. |

---

## 4. Joint map & standing pose

The canonical G1 29-body-joint order (from SAGE-Grasp
`sage_grasp/sim/wholebody/joints.py`) — **note this is *not* the URDF's raw
joint declaration order**, which interleaves the hands:

| Index | Joints | Group | Owner |
|---|---|---|---|
| 0–5 | `left_hip_{pitch,roll,yaw}`, `left_knee`, `left_ankle_{pitch,roll}` | left leg | **AMO** (Stage B) |
| 6–11 | right leg (same pattern) | right leg | **AMO** |
| 12–14 | `waist_{yaw,roll,pitch}` | waist | **AMO** |
| 15–21 | `left_shoulder_{pitch,roll,yaw}`, `left_elbow`, `left_wrist_{roll,pitch,yaw}` | left arm | locked / IK |
| 22–28 | right arm (same pattern) | right arm | **cuRobo** |
| +14 | `{left,right}_hand_{thumb_0/1/2, middle_0/1, index_0/1}` | hands | grasp close |

**AMO standing pose** (`default_dof_pos`): legs `[-0.1, 0, 0, 0.3, -0.2, 0]`,
waist `[0, 0, 0]`, arms `[0.5, 0, ±0.2, 0.3, 0, 0, 0]`. Standing pelvis height
≈ **0.79 m**, `torso_link` ≈ **0.83 m** (waist = 0).

---

## 5. cuRobo config design

The pipeline FKs the **full URDF (rooted at `pelvis`)** for rendering and uses a
**single** `robot_base_T` for both the grasp→base-frame transform and that FK.
So cuRobo's kinematic root must equal the URDF root:

- **`base_link: pelvis`** — matches the URDF/FK root. `robot_base_pose` in the
  robot YAML is therefore the pelvis world pose (`z = 0.787`).
- **Active joints** = the 7 right-arm joints only (`cspace.joint_names`).
- **`lock_joints`** = legs + waist + left arm at the AMO standing pose, and all
  14 hand joints at 0. Locked joints are static in the plan but keep the arm's
  base (torso, via `waist = 0`) at the correct height. Because `waist = 0` in
  both the standing pose *and* the URDFFK render (non-arm joints default to 0),
  the arm-relative-to-pelvis kinematics are consistent between planning and
  rendering regardless of the leg pose.
- **`tool_frames: [right_hand_palm_link]`** — the GraspGenX `unitree_g1` grasp
  frame.
- **Collision spheres** are fitted per link by `build_g1_dex3.py`: each link's
  collision geometry (mesh, or the shoulders' cylinder primitives, or the
  pelvis' visual mesh as a fallback) → an oriented bounding box → a line of
  spheres along its longest axis. Covered links: `pelvis`, `torso_link`, the
  full right-arm chain, and a coarse left-arm stub. Coarse but sufficient for
  arm↔table / arm↔torso avoidance.

Regenerate after any URDF/link change:

```bash
uv run python end2end/build_g1_dex3.py
```

---

## 6. Running Stage A (kinematic pick & lift)

Run from the GraspGenX repo root. Target a free GPU with
`CUDA_VISIBLE_DEVICES`. Requires `uv sync --extra end2end` and the SAGE-Grasp
checkout next to GraspGenX (or `$GRASPGENX_SAGE_DIR`).

```bash
CUDA_VISIBLE_DEVICES=0 PYOPENGL_PLATFORM=egl PYGLET_HEADLESS=true \
uv run python end2end/e2e_grasp_demo.py \
  --robot_config end2end/robots/g1_dex3.yaml \
  --env_config   end2end/envs/g1_tabletop_demo.yaml \
  --task pick_and_lift --playback_mode kinematic --no-viser \
  --num_grasps 200 --topk 80 --grasp_threshold 0.7 --planner graspmoe \
  --hold_after_close_frames 120 --seed 0 \
  --mesh_file assets/sample_data/hope_objects/GranolaBars.obj \
  --export-trajectory end2end/runs/g1_stageA/trajectory.json \
  --render-mp4        end2end/runs/g1_stageA/demo.mp4
```

Export the USD the same way as any other robot:

```bash
PYOPENGL_PLATFORM=egl uv run python end2end/export_trajectory_usd.py \
  --trajectory end2end/runs/g1_stageA/trajectory.json \
  --output     end2end/runs/g1_stageA/demo.usda
```

---

## 7. Stage B plan — AMO lower body in Newton

Goal: the G1 stands on a **floating base** and the **AMO policy** holds balance
(SAGE-Grasp's pelvis-hold standing mode) while cuRobo's right-arm trajectory
executes the pick under Newton physics.

Work items:

1. **Floating base.** Spawn the G1 in `dynamic_playback.py` with a free base
   joint (today the pipeline is fixed-base). The 15 lower-body DOFs (legs +
   waist) become physically actuated rather than locked.
2. **AMO controller in the substep loop.** Port `AMOObservationBuilder` off
   MuJoCo `mjData` onto Newton state: torso orientation / angular velocity /
   height, projected gravity, gait phase, the 10-deep proprio + 25-deep extra
   history buffers, and the adapter network — then run `amo_jit.pt` and write
   the leg+waist PD targets each control step (AMO runs at 50 Hz; decimate vs
   the physics step). The right-arm DOFs stay driven by the cuRobo trajectory
   and the Dex3 fingers by the close logic. `G1Dex3Profile.amo_checkpoints` and
   `lower_body_joint_names` already carry the config this needs.
3. **Consistency.** The right arm's base (`torso_link`) is quasi-static while
   standing, so cuRobo's plan against the locked-leg model stays valid; monitor
   torso drift and re-tune gains if AMO sways.

**Highest risk:** the AMO observation was authored for MuJoCo state conventions
(quaternion order, velocity frames). Porting it to Newton is the crux and needs
careful validation before trusting balance under contact.

---

## 9. Stage A debugging log — what was solved and what remains

The cuRobo grasp plan was debugged in depth; recording it so the next person
doesn't repeat it. Each fix below is real and is in the committed config.

**Solved:**
1. **`ee_link` rejected** — this cuRobo fork's `KinematicsLoaderCfg` doesn't take
   `ee_link`; use `tool_frames` only.
2. **`lock_joints` KeyError** — cuRobo prunes joints not on the chain
   base→{tool, collision links}, so you may only lock joints on those chains.
   The generator now derives the lock set from the URDF chains (waist only, once
   the left-arm collision stub was dropped).
3. **Approach-axis mismatch** — the pipeline assumes approach = tool-frame **+Z**
   (`grasp_approach_axis="z"`), but the Dex3 palm approaches along its **+X**
   (fingers extend +X in `right_hand_palm_link`). Fixed with the `right_dex3_tcp`
   extra-link (rotated so its +Z = the palm's +X); cuRobo drives *that* frame.
4. **Reachability** — the standing right shoulder is at world z≈1.08 m and the arm
   reaches only ~0.45 m; an FK sweep is used to place the object where the palm
   can actually get to it (≥ a few % of sampled configs within 4–6 cm).
5. **Permanent self-collision → `feasible=False` for every pose** (the big one).
   With collisions and limit-clip disabled, cuRobo still marked even the robot's
   own seed pose infeasible. Root cause: coarse OBB spheres on links **2 apart**
   in the chain (e.g. `right_elbow` vs `right_wrist_pitch` overlapped by 5.4 cm;
   the short wrist links nest) — a fixed "immediate-neighbours" ignore list
   misses these. Fixed by **auto-generating the ignore list from the standing FK
   pose** (ignore any pair whose spheres are within 2 cm), plus ignoring the arm
   against its own base body (torso/pelvis), and dropping the left-arm stub
   (locked raised-forward into the workspace). After this, cuRobo's failure mode
   advanced from "Goalset planning returned None" to genuine reachability.

**Remaining:** `plan_grasp` still finds no collision-free plan for the tabletop
scene. IK to the raw grasp poses succeeds only with **both** self-collision and
the table collision off; with either on, or through the full approach→grasp→lift
plan, it fails. The scene is simply tight for a fixed-base humanoid. Likely
next steps, in order:
- **Better collision spheres.** The OBB spheres are coarse. The fork *ships*
  `ext/curobo/curobo/content/configs/robot/unitree_g1.yml` — a validated cuRobo
  G1 sphere set. Basing `collision_spheres` on it (adapting link names / base)
  should remove the remaining false positives without over-ignoring.
- **Ease the scene further / pick object.** Try a shorter object (e.g. a can),
  more table clearance, or the object dead-centre in the reachable orientation
  zone; consider relaxing the approach/lift offsets for the shorter Dex3.
- **Confirm the TCP roll.** The TCP fixes the approach axis but not the roll
  about it; a wrong roll can make otherwise-reachable grasps need an infeasible
  wrist. Validate by visualising one grasp (`--force_grasp_idx`) once a plan
  lands.

## 8. Open issues / tuning knobs

- **`grasp_to_tool_transform`** (`robots/g1_dex3.yaml`) starts at identity on the
  assumption that the GraspGenX `unitree_g1` grasp frame coincides with
  `right_hand_palm_link`. If the closed fingers visibly miss the object,
  calibrate it the way `franka_panda.yaml` documents its +90° Z fix.
- **Dex3 close values** (`G1Dex3Profile.gripper_close`) are coarse finger-curl
  targets within the URDF limits; calibrate against the `unitree_g1`
  `config.json` open/close for a firmer power grasp.
- **Reach.** `envs/g1_tabletop_demo.yaml` shifts the table toward the robot's
  right arm (−y) and closer in x. If cuRobo reports no IK/plan, move the table
  nearer / lower or the object toward the near edge.
- **Left arm render vs plan.** URDFFK renders the (non-arm) left arm at 0 while
  the cuRobo model locks it at the raised standing pose. Harmless for planning
  (object is far from the left arm) but a minor visual mismatch in Stage A.
