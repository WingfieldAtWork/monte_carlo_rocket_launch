"""Guidance flight phase sequencer and attitude target generation.

Flight Phases:
1. Vertical Ascent: t in [0, t_kick_start] -> Point purely vertical.
2. Pitch Kick: t in [t_kick_start, t_kick_start + t_kick_dur] -> Constant pitch rate pitch-over.
3. Gravity Turn: t > t_kick_start + t_kick_dur to MECO -> Align nose with velocity vector v_rel,I.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import numpy as np

from core.coordinates import dcm_to_quat


class FlightPhase(str, Enum):
    VERTICAL_ASCENT = "VERTICAL_ASCENT"
    PITCH_KICK = "PITCH_KICK"
    GRAVITY_TURN = "GRAVITY_TURN"
    COAST = "COAST"


@dataclass
class GuidanceConfig:
    """Guidance parameters."""
    pitch_kick_start_s: float = 5.0
    pitch_kick_duration_s: float = 7.0
    pitch_kick_rate_deg_s: float = 0.10


class GuidanceSystem:
    """Computes target attitude quaternion q_target for the attitude controller."""

    def __init__(self, config: GuidanceConfig):
        self.cfg = config
        self.phase: FlightPhase = FlightPhase.VERTICAL_ASCENT
        self.pitch_kick_rate_rad_s = np.radians(self.cfg.pitch_kick_rate_deg_s)
        self.pitch_kick_end_s = self.cfg.pitch_kick_start_s + self.cfg.pitch_kick_duration_s

        # Initial vertical pitch angle is 90 degrees (pi / 2 rad)
        self.current_pitch_target_rad: float = np.pi / 2.0

    def get_phase(self, t: float, is_burning: bool) -> FlightPhase:
        """Determine current flight phase based on time and engine status."""
        if not is_burning and t > self.pitch_kick_end_s:
            return FlightPhase.COAST
        if t < self.cfg.pitch_kick_start_s:
            return FlightPhase.VERTICAL_ASCENT
        elif t <= self.pitch_kick_end_s:
            return FlightPhase.PITCH_KICK
        else:
            return FlightPhase.GRAVITY_TURN

    def update(
        self,
        t: float,
        pos_i: np.ndarray,
        v_rel_i: np.ndarray,
        is_burning: bool = True,
    ) -> tuple[np.ndarray, FlightPhase]:
        """Compute target attitude quaternion q_target(t).

        Inertial axes convention:
        - +z_I: Vertical Up at launch site
        - +x_I: Downrange
        - +y_I: Crossrange

        Body axes convention:
        - +x_B: Long axis (nose)
        - +y_B: Lateral axis (pitch axis)
        - +z_B: Normal axis (yaw axis)

        At vertical launch:
        +x_B points along +z_I
        +y_B points along +y_I
        +z_B points along -x_I

        Args:
            t: Current simulation time (s).
            pos_i: Position vector in inertial frame (m).
            v_rel_i: Relative velocity vector in inertial frame (m/s).
            is_burning: True if propulsion is active.

        Returns:
            q_target: Target quaternion q_I^B (scalar-first).
            phase: Current FlightPhase enum.
        """
        self.phase = self.get_phase(t, is_burning)

        if self.phase == FlightPhase.VERTICAL_ASCENT:
            # Pure vertical ascent: pitch = 90 deg
            self.current_pitch_target_rad = np.pi / 2.0
            r_b_i = np.array([
                [0.0, 0.0, -1.0],
                [0.0, 1.0,  0.0],
                [1.0, 0.0,  0.0],
            ], dtype=np.float64)

        elif self.phase == FlightPhase.PITCH_KICK:
            # Constant pitch-over rate towards downrange (+x_I)
            dt_kick = t - self.cfg.pitch_kick_start_s
            pitch_angle = np.pi / 2.0 - self.pitch_kick_rate_rad_s * dt_kick
            self.current_pitch_target_rad = pitch_angle

            # Target nose vector u_target = [cos(theta), 0, sin(theta)]
            c_th = np.cos(pitch_angle)
            s_th = np.sin(pitch_angle)

            r_b_i = np.array([
                [c_th, 0.0, -s_th],
                [0.0,  1.0,   0.0],
                [s_th, 0.0,  c_th],
            ], dtype=np.float64)

        else:
            # Gravity turn or coast: align vehicle nose (+x_B) with relative velocity vector
            v_norm = float(np.linalg.norm(v_rel_i))

            if v_norm > 5.0:
                u_target = v_rel_i / v_norm

                # Construct orthonormal frame:
                # Keep lateral body axis y_B horizontal (parallel to crossrange / perpendicular to velocity and vertical)
                # Local vertical direction
                r_ec = np.array([pos_i[0], pos_i[1], pos_i[2] + 6378137.0], dtype=np.float64)
                up_vec = r_ec / np.linalg.norm(r_ec)

                y_temp = np.cross(up_vec, u_target)
                y_norm = float(np.linalg.norm(y_temp))

                if y_norm > 1e-4:
                    y_target = y_temp / y_norm
                else:
                    y_target = np.array([0.0, 1.0, 0.0], dtype=np.float64)

                z_target = np.cross(u_target, y_target)

                # Body to Inertial DCM: columns are [x_target, y_target, z_target]
                r_b_i = np.column_stack([u_target, y_target, z_target])
            else:
                # If speed is too low, hold end-of-kick attitude
                c_th = np.cos(self.current_pitch_target_rad)
                s_th = np.sin(self.current_pitch_target_rad)
                r_b_i = np.array([
                    [c_th, 0.0, -s_th],
                    [0.0,  1.0,   0.0],
                    [s_th, 0.0,  c_th],
                ], dtype=np.float64)

        # Convert R_B^I to R_I^B = (R_B^I)^T
        r_i_b = r_b_i.T
        q_target = dcm_to_quat(r_i_b)
        return q_target, self.phase

