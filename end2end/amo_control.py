"""Vendored AMO whole-body *balance* controller for the Unitree G1 (Stage B).

Reimplemented from SAGE-Grasp ``sage_grasp/sim/wholebody/control_stack.py``
(``AMOObservationBuilder`` + ``G1AMOBalanceController``) so GraspGenX has **no
SAGE dependency** — it only needs the three vendored TorchScript / stats files
under ``end2end/robots/g1/amo_policy/`` (``amo_jit.pt``, ``adapter_jit.pt``,
``adapter_norm_stats.pt``).

The observation layout MUST stay byte-for-byte identical to what the policy was
trained with, so this is a faithful port — do not "clean up" the magic indices.

Newton-agnostic: :meth:`AMOBalanceController.lower_body_targets` takes plain
numpy state (29 body-joint positions/velocities in the canonical G1 order, base
orientation + angular velocity + height) and returns the 15 lower-body
(legs+waist) **position** targets. The caller feeds those into Newton's
``joint_target_pos`` (POSITION drive mode) at the AMO control rate (50 Hz); the
arms are driven separately (right arm from the cuRobo trajectory, left arm held).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Canonical G1 29-body-joint order (SAGE joints.py G1_JOINT_NAME_MAP). The AMO
# obs/action are indexed in THIS order — it is *not* the URDF's raw joint order.
G1_BODY_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
# The 15 lower-body joints AMO owns (legs 0-11 + waist 12-14).
LOWER_BODY_SLICE = slice(0, 15)

# AMO standing / default pose (29 body joints, canonical order). Same values the
# policy normalises against; used to seed the Newton state and as the PD offset.
AMO_DEFAULT_DOF_POS = np.array(
    [
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        0.0, 0.0, 0.0,
        0.5, 0.0, 0.2, 0.3, 0.0, 0.0, 0.0,
        0.5, 0.0, -0.2, 0.3, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)


def quaternion_wxyz_to_rpy(quat: np.ndarray) -> np.ndarray:
    """(w,x,y,z) unit-ish quaternion -> (roll, pitch, yaw). Matches SAGE."""
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        return np.zeros(3)
    w, x, y, z = quat / norm
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float64)


class AMOObservationBuilder:
    """Build the observation the AMO teleop policy expects (faithful SAGE port)."""

    def __init__(
        self,
        adapter_path: str | Path,
        norm_stats_path: str | Path,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.adapter = torch.jit.load(str(adapter_path), map_location=self.device)
        self.adapter.eval()
        for p in self.adapter.parameters():
            p.requires_grad = False

        norm_stats = torch.load(str(norm_stats_path), weights_only=False)
        self.input_mean = torch.tensor(norm_stats["input_mean"], device=self.device, dtype=torch.float32)
        self.input_std = torch.tensor(norm_stats["input_std"], device=self.device, dtype=torch.float32)
        self.output_mean = torch.tensor(norm_stats["output_mean"], device=self.device, dtype=torch.float32)
        self.output_std = torch.tensor(norm_stats["output_std"], device=self.device, dtype=torch.float32)

        self.scales_ang_vel = 0.25
        self.scales_dof_vel = 0.05
        self.n_priv = 3
        self.n_proprio = 3 + 2 + 2 + 23 * 3 + 2 + 15  # = 93
        self.history_len = 10
        self.extra_history_len = 25
        self.n_demo_dof = 8
        self.action_scale = 0.25

        # Default standing pose (== AMO_STANDING in build_g1_dex3.py).
        self.default_dof_pos = AMO_DEFAULT_DOF_POS.copy()

        self.last_action = np.zeros(29, dtype=np.float64)
        self.demo_obs_template = np.zeros((8 + 3 + 3 + 3,), dtype=np.float64)
        self.demo_obs_template[: self.n_demo_dof] = self.default_dof_pos[np.r_[15:19, 22:26]]
        self.demo_obs_template[self.n_demo_dof + 6 : self.n_demo_dof + 9] = 0.75

        self.target_yaw = 0.0
        self.dyaw = 0.0
        self.dt = 0.02
        self.gait_cycle = np.array([0.25, 0.25], dtype=np.float64)
        self.gait_freq = 1.3
        self.control_dt = 0.02

        self.proprio_history_buf = deque(maxlen=self.history_len)
        self.extra_history_buf = deque(maxlen=self.extra_history_len)
        self.reset()

    def reset(self) -> None:
        self.last_action = np.zeros(29, dtype=np.float64)
        self.target_yaw = 0.0
        self.gait_cycle = np.array([0.25, 0.25], dtype=np.float64)
        self.proprio_history_buf.clear()
        self.extra_history_buf.clear()
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))
        for _ in range(self.extra_history_len):
            self.extra_history_buf.append(np.zeros(self.n_proprio))

    def build(
        self,
        motorstate: np.ndarray,
        velstate: np.ndarray,
        rpy: np.ndarray,
        ang_vel: np.ndarray,
        *,
        torso_height: float = 0.75,
        torso_yaw: float = 0.0,
        torso_pitch: float = 0.0,
        torso_roll: float = 0.0,
        vx: float = 0.0,
        vy: float = 0.0,
        vyaw: float = 0.0,
        yaw_offset: float = 0.0,
    ) -> tuple[np.ndarray, deque]:
        rpy = np.asarray(rpy, dtype=np.float64).reshape(3)
        ang_vel = np.asarray(ang_vel, dtype=np.float64).reshape(3)

        self.target_yaw += vyaw * self.dt
        dyaw = rpy[2] - yaw_offset - self.target_yaw
        dyaw = np.remainder(dyaw + np.pi, 2 * np.pi) - np.pi
        in_place = abs(vx) < 0.1 and abs(vy) < 0.1 and abs(vyaw) < 0.1
        if in_place:
            dyaw = 0.0
        self.dyaw = float(dyaw)

        obs_idx = np.r_[0:19, 22:26]
        obs_dof_vel = velstate[obs_idx].copy()
        obs_dof_vel[[4, 5, 10, 11, 13, 14]] = 0.0
        obs_dof_pos = motorstate[obs_idx].copy()
        obs_default_dof_pos = self.default_dof_pos[obs_idx]
        obs_last_action = self.last_action[obs_idx]

        gait_obs = np.sin(self.gait_cycle * 2 * np.pi)

        adapter_input_np = np.concatenate([np.zeros(4), obs_dof_pos[15:]])
        adapter_input_np[0] = torso_height
        adapter_input_np[1] = torso_yaw
        adapter_input_np[2] = torso_pitch
        adapter_input_np[3] = torso_roll

        adapter_input = torch.tensor(adapter_input_np, device=self.device, dtype=torch.float32).unsqueeze(0)
        adapter_input = (adapter_input - self.input_mean) / (self.input_std + 1e-8)
        adapter_output = self.adapter(adapter_input.view(1, -1))
        adapter_output = adapter_output * self.output_std + self.output_mean

        obs_prop = np.concatenate(
            [
                ang_vel * self.scales_ang_vel,
                rpy[:2],
                (np.sin(self.dyaw), np.cos(self.dyaw)),
                obs_dof_pos - obs_default_dof_pos,
                obs_dof_vel * self.scales_dof_vel,
                obs_last_action,
                gait_obs,
                adapter_output.detach().cpu().numpy().squeeze(),
            ]
        )

        obs_priv = np.zeros((self.n_priv,))
        obs_hist = np.array(self.proprio_history_buf).flatten()

        obs_demo = self.demo_obs_template.copy()
        obs_demo[: self.n_demo_dof] = obs_dof_pos[15:]
        obs_demo[self.n_demo_dof] = vx
        obs_demo[self.n_demo_dof + 1] = vy
        obs_demo[self.n_demo_dof + 3] = torso_yaw
        obs_demo[self.n_demo_dof + 4] = torso_pitch
        obs_demo[self.n_demo_dof + 5] = torso_roll
        obs_demo[self.n_demo_dof + 6 : self.n_demo_dof + 9] = torso_height

        self.proprio_history_buf.append(obs_prop)
        self.extra_history_buf.append(obs_prop)
        observation = np.concatenate((obs_prop, obs_demo, obs_priv, obs_hist))

        self.gait_cycle = np.remainder(self.gait_cycle + self.control_dt * self.gait_freq, 1.0)
        if in_place and (abs(self.gait_cycle[0] - 0.25) < 0.05 or abs(self.gait_cycle[1] - 0.25) < 0.05):
            self.gait_cycle = np.array([0.25, 0.25])
        if (not in_place) and (abs(self.gait_cycle[0] - 0.25) < 0.05 and abs(self.gait_cycle[1] - 0.25) < 0.05):
            self.gait_cycle = np.array([0.25, 0.75])

        return observation, self.extra_history_buf

    def update_last_action(self, raw_action: np.ndarray, motorstate: np.ndarray) -> None:
        self.last_action = np.concatenate(
            [raw_action.copy(), (motorstate - self.default_dof_pos)[15:] / self.action_scale]
        )


class AMOBalanceController:
    """AMO lower-body balance policy (legs + waist), Newton-agnostic.

    ``lower_body_targets`` runs the policy once (call it at the 50 Hz AMO control
    rate) and returns the 15 leg+waist **position** targets to feed into Newton's
    ``joint_target_pos``. The arms are NOT touched here — the caller holds/drives
    them separately.
    """

    def __init__(
        self,
        amo_policy_path: str | Path,
        adapter_path: str | Path,
        norm_stats_path: str | Path,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            raise RuntimeError(
                "The AMO TorchScript policy is CUDA-bound (its graph allocates "
                "cuda tensors internally). Run Stage B on a CUDA device."
            )
        self.policy = torch.jit.load(str(amo_policy_path), map_location=self.device)
        self.policy.eval()
        self.obs = AMOObservationBuilder(adapter_path, norm_stats_path, device=self.device)
        self.default_dof_pos = self.obs.default_dof_pos
        self.action_scale = self.obs.action_scale
        self.control_dt = self.obs.control_dt
        self.last_raw_action = np.zeros(15, dtype=np.float64)

    def reset(self) -> None:
        self.obs.reset()
        self.last_raw_action = np.zeros(15, dtype=np.float64)

    def lower_body_targets(
        self,
        q29: np.ndarray,
        dq29: np.ndarray,
        base_quat_wxyz: np.ndarray,
        base_ang_vel: np.ndarray,
        base_height: float,
    ) -> np.ndarray:
        """Return the 15 leg+waist position targets for this control step.

        q29 / dq29: body joint pos/vel in the canonical G1 order (G1_BODY_JOINTS).
        base_quat_wxyz: pelvis orientation as (w,x,y,z).
        base_ang_vel: pelvis angular velocity in the world/base frame (3).
        base_height: pelvis height (world z).
        """
        q29 = np.asarray(q29, dtype=np.float64).reshape(29)
        dq29 = np.asarray(dq29, dtype=np.float64).reshape(29)
        rpy = quaternion_wxyz_to_rpy(base_quat_wxyz)

        observation, extra_hist = self.obs.build(
            q29, dq29, rpy, base_ang_vel,
            torso_height=float(base_height),
            torso_yaw=float(rpy[2]), torso_pitch=float(rpy[1]), torso_roll=float(rpy[0]),
            vx=0.0, vy=0.0, vyaw=0.0,  # standing balance: zero command velocity
        )

        obs_t = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        extra_t = torch.tensor(
            np.array(extra_hist).flatten().copy(), dtype=torch.float32, device=self.device
        ).view(1, -1)
        with torch.no_grad():
            raw_action = self.policy(obs_t, extra_t).cpu().numpy().squeeze()
        raw_action = np.clip(raw_action, -40.0, 40.0)

        targets = self.default_dof_pos[LOWER_BODY_SLICE] + raw_action * self.action_scale
        self.last_raw_action = raw_action.copy()
        self.obs.update_last_action(raw_action, q29)
        return targets.astype(np.float64)
