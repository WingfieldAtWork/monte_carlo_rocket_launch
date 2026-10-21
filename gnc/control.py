"""Attitude control module: quaternion-error PD controller for TVC gimbal."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from core.coordinates import quat_conjugate, quat_multiply


@dataclass
class ControlConfig:
    """Attitude controller gains and limits."""
    control_kp: float = 8.0
    control_kd: float = 3.5
    gimbal_limit_deg: float = 6.0


@dataclass
class ControlOutput:
    """Controller calculation outputs."""
    q_err: np.ndarray       # Error quaternion [q0, q1, q2, q3]
    error_vector: np.ndarray  # Rotation error vector e [ex, ey, ez] (rad)
    control_u: np.ndarray   # Raw PD control vector u [ux, uy, uz]
    delta_y_cmd: float      # Commanded pitch gimbal angle (rad)
    delta_z_cmd: float      # Commanded yaw gimbal angle (rad)


class AttitudeController:
    """Quaternion-error PD attitude controller for TVC gimbal steering."""

    def __init__(self, config: ControlConfig):
        self.cfg = config
        self.gimbal_limit_rad = np.radians(self.cfg.gimbal_limit_deg)

    def compute_control(
        self,
        q_current: np.ndarray,
        q_target: np.ndarray,
        omega_b: np.ndarray,
    ) -> ControlOutput:
        """Compute TVC gimbal commands from attitude error.

        1. Error quaternion:
           q_err = q^-1 (x) q_target
        2. Rotation error vector:
           e = 2 * sgn(q_err_0) * [q_err_1, q_err_2, q_err_3]^T
        3. PD control effort:
           u = Kp * e - Kd * omega_B
        4. TVC gimbal command mapping:
           delta_y_cmd = -u_y
           delta_z_cmd = -u_z

        Args:
            q_current: Current attitude quaternion q_I^B (scalar-first).
            q_target: Target attitude quaternion q_I^B (scalar-first).
            omega_b: Current body angular rates [wx, wy, wz] (rad/s).

        Returns:
            ControlOutput dataclass.
        """
        # 1. Error quaternion: q_err = q^-1 (x) q_target
        q_inv = quat_conjugate(q_current)
        q_err = quat_multiply(q_inv, q_target)

        # 2. Extract rotation error vector:
        # e = 2 * sgn(q_err,0) * [q_err,1, q_err,2, q_err,3]^T
        sign_q0 = 1.0 if q_err[0] >= 0.0 else -1.0
        error_vector = 2.0 * sign_q0 * q_err[1:4]

        # 3. Controller output: u = K_p * e - K_d * omega_B
        control_u = self.cfg.control_kp * error_vector - self.cfg.control_kd * omega_b

        # 4. Map control effort to TVC gimbal deflections:
        # Pitch TVC: delta_y produces negative pitch moment M_y = -x_CG * T * sin(delta_y)
        # Therefore delta_y_cmd = -u_y
        delta_y_cmd = float(np.clip(-control_u[1], -self.gimbal_limit_rad, self.gimbal_limit_rad))

        # Yaw TVC: delta_z produces negative yaw moment M_z = -x_CG * T * sin(delta_z)
        # Therefore delta_z_cmd = -u_z
        delta_z_cmd = float(np.clip(-control_u[2], -self.gimbal_limit_rad, self.gimbal_limit_rad))

        return ControlOutput(
            q_err=q_err,
            error_vector=error_vector,
            control_u=control_u,
            delta_y_cmd=delta_y_cmd,
            delta_z_cmd=delta_z_cmd,
        )

