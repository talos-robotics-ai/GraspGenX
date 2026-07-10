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
| **A** | Mirror the Franka structure: robot/env YAML + profile + cuRobo config; G1 loads, GraspGenX predicts Dex3 grasps, cuRobo plans the right arm, **kinematic** playback + MP4/USD export. Fixed base. | **Working** — cuRobo plans the full approach→grasp→lift; a 722-frame `trajectory.json` exports and the USD validates. Run cmd in §6. |
| **B** | Floating base in Newton + AMO lower-body controller in the `dynamic_playback` substep loop; the G1 balances while the arm executes the pick. | **Planned** (design in §7) |

Stage A changes nothing about how a robot is selected or imported — it is a new
`--robot_config` YAML exactly like `franka_panda.yaml`, and **no existing
pipeline file was modified** (only `paths.py` gained a `${G1}` token and
`robot_profiles.py` gained the `G1Dex3Profile`). The G1 + Dex3 assets (URDF,
meshes, AMO checkpoints) are **vendored in-repo** under `end2end/robots/g1/` —
no external SAGE-Grasp checkout is needed.

**What is verified working (§9 has the detail):**
- The G1 + Dex3 loads; the robot/env/profile wiring resolves end-to-end
  (URDF FK over all 43 joints, `${G1}`/`${E2E}` expansion, cuRobo placeholder
  substitution).
- **GraspGenX predicts real Dex3 grasps** (gripper `unitree_g1`, 80 grasps,
  confidence 0.97–0.99).
- The cuRobo config is **valid and self-consistent**: base=`pelvis`, the 7
  right-arm joints active, the rest locked at standing, a synthetic **TCP frame**
  fixing the Dex3 approach axis, and an **FK-derived self-collision ignore list**
  so the model is no longer permanently self-colliding.

**What made it work (the last two pieces):**
- **Validated shipped spheres** (§9.6) make the grasp poses IK-reachable
  (`feasible=True`).
- A **palm-over-object "ready" start pose** (`RIGHT_ARM_READY`, now the default)
  gives the trajectory optimiser a short path — so cuRobo plans the full
  approach→grasp→lift.
- A **one-line cuRobo-fork fix** (§9.8): the fork crashed reindexing a joint
  state whose `knot` tensor is sized to the active DOF while `position` includes
  the locked joints. `reindex_joint_state_inplace` now skips fields whose last
  dim doesn't match the full joint count. Needed because the G1 config **locks
  joints** (the waist) — franka locks none, so it never hit this.

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
| `paths.py` | +`${G1}` token (`g1_assets_dir()`) | Resolve the vendored G1 assets under `end2end/robots/g1/` (URDF, meshes, AMO checkpoints) — self-contained, no external checkout. |
| `robots/g1/` | **new** — vendored G1 + Dex3 URDF, meshes, `amo_policy/` checkpoints | Copied in-repo so the pipeline has no SAGE-Grasp dependency. |
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
- **`cspace.default_joint_position` = `RIGHT_ARM_READY`** — the plan start/seed,
  an FK-found pose with the palm hovering over the object pointing down. This is
  what lets the trajectory optimiser converge (a raised standing seed does not).
- **`lock_joints`** = only the joints on cuRobo's retained chains that aren't
  active — i.e. the 3 **waist** joints (they lie between `pelvis` and the arm).
  The generator derives this set from the URDF chains; legs / left arm / hands
  are pruned by cuRobo (not on any retained chain) so they must **not** be
  locked. **Note:** locking joints is what exposed the fork bug fixed in §9.8.
- **`tool_frames: [right_dex3_tcp]`** — the synthetic TCP extra-link whose +Z is
  the Dex3 palm's real approach axis (§9.3).
- **Collision spheres** come from cuRobo's shipped `unitree_g1.yml` (dense,
  validated; `_shipped_g1_spheres()`), with the per-link OBB fitter as a
  fallback. Covered links: `pelvis`, `torso_link`, the full right-arm chain.
  `self_collision_ignore` is auto-generated by FK-ing the standing pose and
  ignoring pairs whose spheres overlap at rest, plus the arm vs its base body.

Regenerate after any URDF/link change:

```bash
uv run python end2end/build_g1_dex3.py
```

---

## 6. Running Stage A (kinematic pick & lift)

Run from the GraspGenX repo root. Target a free GPU with
`CUDA_VISIBLE_DEVICES`. Requires `uv sync --extra end2end`. The G1 + Dex3 assets
are vendored under `end2end/robots/g1/` — no external checkout needed.

**This exact command lands a plan** (722-frame trajectory) — verified. The
`--max_plan_attempts` matched to `--topk` is required: cuRobo's goalset index can
otherwise overrun the (smaller) goalset. `--render-mp4` appends the MP4 render to
the same run.

```bash
CUDA_VISIBLE_DEVICES=0 PYOPENGL_PLATFORM=egl PYGLET_HEADLESS=true \
uv run python end2end/e2e_grasp_demo.py \
  --robot_config end2end/robots/g1_dex3.yaml \
  --env_config   end2end/envs/g1_tabletop_demo.yaml \
  --task pick_and_lift --playback_mode kinematic --no-viser \
  --num_grasps 400 --topk 100 --grasp_threshold 0.5 --planner graspmoe \
  --max_plan_attempts 100 --hold_after_close_frames 120 --seed 0 \
  --grasp_overlay_in_mp4 chosen \
  --mesh_file assets/sample_data/hope_objects/GranolaBars.obj \
  --export-trajectory end2end/runs/g1_stageA/trajectory.json \
  --render-mp4        end2end/runs/g1_stageA/demo.mp4
```

`--grasp_overlay_in_mp4 chosen` draws only the picked grasp triad on the box
(clean view). The default `all` also draws the full 100-candidate cloud, whose
frames sit ~1 finger-length *off* the object (the GraspGen grasp frame is at the
hand base, not the contact patch), so it reads as a cloud offset from the box —
expected, not a placement bug. Use `none` to hide the overlay entirely.

In **kinematic** mode the exporter now rigidly **attaches the object to the tool
frame once the fingers close** (end of the `close_fingers` segment), so
`pick_and_lift` actually shows the box lifting with the hand (it uses the same
`objects` + per-frame `object_poses` schema as the dynamic path). Before this,
kinematic mode pinned the object to the table and the lift looked like a slipped
grasp. See §9.10.

Export the USD the same way as any other robot (renders in Isaac Sim / usdview;
a sibling `textures/` folder is written next to the `.usda`):

```bash
PYOPENGL_PLATFORM=egl uv run python end2end/export_trajectory_usd.py \
  --trajectory end2end/runs/g1_stageA/trajectory.json \
  --output     end2end/runs/g1_stageA/demo.usda
```

The MP4 render (`render_trajectory_mp4.py`) can also be run standalone on an
existing `trajectory.json`; add `--resolution 640x480` etc. as needed.

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

6. **Coarse spheres → validated shipped spheres** (the fix for #5's leftovers).
   The fork *ships* a proper cuRobo G1 config,
   `ext/curobo/curobo/content/configs/robot/unitree_g1.yml`, with dense hand-tuned
   `collision_spheres` under the **same link names** as the SAGE URDF. The
   generator now pulls those per link (OBB only as a fallback;
   `_shipped_g1_spheres()`). With them, **the false self-collisions vanish and the
   grasp poses become IK-reachable — `feasible=True`, 0 pose error** — confirmed
   across self/table on×off. So collision is no longer the blocker.
7. **Object placement at the workspace sweet spot.** An FK map of the right
   arm's reachability (`build`-style sampling) put the object where the palm has
   maximum orientation freedom (close to the body, x≈0.20; moderately high,
   z≈0.82) rather than at the reach edge. Table-height sweeps (0.60–0.82 m)
   showed height is *not* the lever — reachability/orientation is.

8. **`plan_pose` CUDA assert with locked joints → cuRobo-fork fix (the last
   blocker).** Two sub-parts:
   - **Start pose.** From the raised **standing** seed, trajopt doesn't converge
     (returns None). From a **"ready"** seed — right palm hovering over the
     object pointing down (`RIGHT_ARM_READY`, FK-found, now the
     `default_joint_position`) — cuRobo plans the full approach→grasp→lift.
   - **The fork bug.** With the ready seed, `plan_grasp` progressed past IK and
     then crashed with a device-side CUDA assert
     `indexSelectSmallIndex: srcIndex < srcSelectDimSize`. Traced (with
     `CUDA_LAUNCH_BLOCKING=1` + surfacing the swallowed traceback) to
     `state_joint_ops.reindex_joint_state_inplace` →
     `state_joint_jit_helpers.jit_inplace_reindex`. Root cause: for a robot with
     **locked joints**, a `JointState`'s `position/velocity/...` carry all joints
     (here 10 = 3 locked waist + 7 active) but the `knot` tensor is sized to the
     **active DOF only** (7); reindexing applied the full-joint index (values up
     to 9) to the 7-wide `knot` → out of bounds. Franka locks no joints, so it
     never hit this. **Fix (in `ext/curobo/.../state/state_joint_ops.py`):**
     `reindex_joint_state_inplace` now reindexes only fields whose last dim
     matches the full joint count and leaves mismatched ones (e.g. `knot`)
     untouched. One-file, backward-compatible change; robots without locked
     joints are unaffected. **Caveat:** it lives in the `ext/curobo` checkout, so
     re-cloning cuRobo (`setup_end2end_deps.py`) will drop it — re-apply, or
     upstream it.
   - `--max_plan_attempts` must equal `--topk` (see §6): otherwise cuRobo's
     goalset index can overrun the smaller goalset (a separate size mismatch).

With all of the above, **Stage A plans and exports end-to-end** — a 722-frame
`trajectory.json` and a validated USD.

Follow-ups (nice-to-have, not blockers):
- **Confirm the TCP roll** (`--force_grasp_idx`): the TCP fixes the approach axis
  but not the roll about it; check the closed fingers wrap the object as intended.
- **A nicer start pose.** `RIGHT_ARM_READY` is a functional (slightly awkward)
  hover; a hand-tuned ready pose would make the render prettier.

9. **Three render-stage bugs found from the first MP4 (all fixed).** The plan
   landed but the first video showed a garbage arm, no object, and self-collision.
   Root causes:
   - **Locked joints corrupt the arm trajectory (the big one).** cuRobo's
     trajectory columns are all cspace DOFs in *kinematic order*, so the G1's
     locked **waist precedes the arm**; the extractors sliced `[:n_arm]` and got
     `[waist, waist, waist, arm0..3]` → a shifted, nonsensical arm that never
     reached the object and appeared to self-collide. **Fix:** select the
     active-arm columns *by name* (`e2e_grasp_demo._traj_to_np` and
     `tasks._plan_arm_to_pose`), using the trajectory's `joint_names`. Franka's
     arm is first, so it's a no-op there — but this is a **general** pipeline fix
     for any robot with locked joints ahead of the arm.
   - **Object invisible (wrong scale).** `scene_builder` built the object's vis
     mesh from a raw `synth.MeshAsset(mesh_file)`, ignoring the env `mesh_scale`
     — HOPE meshes are millimetres, so it rendered ~1000× too big. **Fix:**
     `vis_meshes["object"]` now uses the already scaled+rotated `object_mesh`.
   - **Self-collision.** The blanket "ignore torso/pelvis vs the whole arm" (a
     workaround for the earlier *coarse* OBB spheres) let cuRobo route the arm
     through the trunk. **Fix:** removed it — with the validated shipped spheres
     the torso↔forearm/hand checks are accurate (only the true rest-pose
     adjacencies, e.g. shoulder↔torso, are ignored). Verified: min self-collision
     sphere separation over the whole trajectory is **+0.02 m** (no overlaps).

   After these, the render shows a clean hover→reach-down→grasp→lift on the
   (correctly sized) GranolaBars box with no self-collision.

10. **Second MP4 review — "grasp inconsistent with the box; collision detection
    doesn't work" (all fixed).** The plan was collision-free in cuRobo's model,
    but the video still looked wrong. Three findings, all addressed:
    - **The object was placed too close/inboard (root cause).** It sat at world
      `x≈0.20`, i.e. 8 cm *behind* where the arm naturally reaches and only ~4 cm
      off the torso front — so the whole grasp happened jammed against the belly.
      A DLS-IK reach sweep of the fixed-base standing G1 showed the right palm
      reaches top-down grasps out to `x≈0.28` across the whole y-range, and the
      `RIGHT_ARM_READY` seed already hovers at world `[0.273, −0.158, 0.902]`.
      **Fix:** place the object directly under that reach point — table centred on
      `y=−0.15`, object at `x≈0.28` (`translation_offset` `[-0.25,0,0]`). The box
      front face now clears the torso (front at `x≈0.08`) by ~12 cm, and every
      distal arm link clears the trunk with margin (measured over all 722 frames:
      elbow **+0.15 m**, wrist **+0.16–0.19 m**, hand/palm **+0.19 m** from the
      torso). The only sub-zero pair is `right_shoulder_pitch_link` vs `torso_link`
      (−9 mm) — the shoulder is *bolted onto* the torso, so it's on cuRobo's
      ignore list. "Collision detection doesn't work" was really "the meshes
      visually touch because the grasp is crammed against the body, even though
      the sphere model has +2 cm clearance"; roomier placement resolves it.
    - **The candidate-grasp cloud reads as "inconsistent with the box".** The
      renderer drew all 100 GraspGen candidates; their frames are at the hand
      *base* (~13 cm off the object by finger reach), so front-approach candidates
      have origins out at `x≈0.43` and project to a cluster offset from the box.
      Not a bug, but confusing. **Fix:** new `--grasp-overlay {all,chosen,none}`
      (renderer) / `--grasp_overlay_in_mp4` (e2e driver). The G1 command uses
      `chosen` → only the picked grasp triad, which is dead-centred on the box
      (offset `[0.002, 0, 0.122]` = a clean 12 cm top approach). Default stays
      `all` for the franka demos.
    - **In kinematic mode the object stayed pinned to the table during the lift.**
      That's existing kinematic-export behaviour (only the *dynamic* path carried
      the object), so the "lift" looked like a slipped grasp. **Fix:**
      `export_trajectory` now rigidly attaches the object to the tool frame from
      the end of `close_fingers` onward and emits it via the `objects` +
      per-frame `object_poses` schema (same as dynamic; the renderer and USD
      exporter already consume it, and the HOPE texture is preserved). Verified:
      the box holds still through approach/close, then lifts **0.20 m** with the
      hand, rigid-attach drift ~1 mm.

    After these the render shows the box out on the table, a single grasp triad on
    it, the right arm reaching in free space (clear of the torso), and the box
    lifting with the hand. Run dir: `end2end/runs/g1_stageA/`.

## 8. Open issues / tuning knobs

- **`grasp_to_tool_transform`** (`robots/g1_dex3.yaml`) starts at identity on the
  assumption that the GraspGenX `unitree_g1` grasp frame coincides with
  `right_hand_palm_link`. If the closed fingers visibly miss the object,
  calibrate it the way `franka_panda.yaml` documents its +90° Z fix.
- **Dex3 close values** (`G1Dex3Profile.gripper_close`) are coarse finger-curl
  targets within the URDF limits; calibrate against the `unitree_g1`
  `config.json` open/close for a firmer power grasp.
- **Reach (fixed-base limit).** With the waist locked (Stage A), the right palm
  only reaches top-down tabletop grasps out to `x≈0.28`; the object is placed
  right there, under the `RIGHT_ARM_READY` hover point (§9.10). A normal reach-out
  grasp (`x≈0.4–0.5`) needs the torso to lean, which is Stage B (AMO). If cuRobo
  reports no IK/plan, keep the object in `x∈[0.22, 0.30]`, `y∈[−0.20, −0.10]`.
- **Left arm render vs plan.** URDFFK renders the (non-arm) left arm at 0 while
  the cuRobo model locks it at the raised standing pose. Harmless for planning
  (object is far from the left arm) but a minor visual mismatch in Stage A.
