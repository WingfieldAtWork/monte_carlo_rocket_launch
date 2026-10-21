"""Quaternion mathematics, Direction Cosine Matrices (DCM), and frame transformations.

Conventions:
- Scalar-first quaternion: q = [q0, q1, q2, q3]^T where q0 is scalar and [q1, q2, q3] is vector part.
- q represents attitude from Inertial to Body frame (q_I^B).
- R_B^I is DCM from Body to Inertial frame: v_I = R_B^I @ v_B.
- R_I^B is DCM from Inertial to Body frame: v_B = R_I^B @ v_I = (R_B^I)^T @ v_I.
"""

from __future__ import annotations
import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize a quaternion to unit length.

    Args:
        q: Array of shape (4,) representing [q0, q1, q2, q3].

    Returns:
        Unit quaternion of shape (4,).
    """
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Compute the quaternion conjugate (inverse for unit quaternions).

    Args:
        q: Quaternion [q0, q1, q2, q3].

    Returns:
        Conjugate quaternion [q0, -q1, -q2, -q3].
    """
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two scalar-first quaternions: q = q1 (x) q2.

    Args:
        q1: Left quaternion [w1, x1, y1, z1].
        q2: Right quaternion [w2, x2, y2, z2].

    Returns:
        Product quaternion [w, x, y, z].
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z], dtype=np.float64)


def quat_to_dcm_ib(q: np.ndarray) -> np.ndarray:
    """Compute Direction Cosine Matrix R_I^B (Inertial to Body) from quaternion q_I^B.

    Args:
        q: Quaternion [q0, q1, q2, q3] mapping Inertial to Body.

    Returns:
        3x3 rotation matrix R_I^B such that v_B = R_I^B @ v_I.
    """
    q0, q1, q2, q3 = quat_normalize(q)

    q0_2 = q0 * q0
    q1_2 = q1 * q1
    q2_2 = q2 * q2
    q3_2 = q3 * q3

    return np.array(
        [
            [
                q0_2 + q1_2 - q2_2 - q3_2,
                2.0 * (q1 * q2 + q0 * q3),
                2.0 * (q1 * q3 - q0 * q2),
            ],
            [
                2.0 * (q1 * q2 - q0 * q3),
                q0_2 - q1_2 + q2_2 - q3_2,
                2.0 * (q2 * q3 + q0 * q1),
            ],
            [
                2.0 * (q1 * q3 + q0 * q2),
                2.0 * (q2 * q3 - q0 * q1),
                q0_2 - q1_2 - q2_2 + q3_2,
            ],
        ],
        dtype=np.float64,
    )


def quat_to_dcm_bi(q: np.ndarray) -> np.ndarray:
    """Compute Direction Cosine Matrix R_B^I (Body to Inertial) from quaternion q_I^B.

    Args:
        q: Quaternion [q0, q1, q2, q3] mapping Inertial to Body.

    Returns:
        3x3 rotation matrix R_B^I such that v_I = R_B^I @ v_B.
    """
    return quat_to_dcm_ib(q).T


def dcm_to_quat(dcm: np.ndarray) -> np.ndarray:
    """Convert 3x3 Direction Cosine Matrix R_I^B to scalar-first quaternion q_I^B.

    Uses Shepperd's robust algorithm to avoid division by zero or numerical instability.

    Args:
        dcm: 3x3 orthonormal rotation matrix R_I^B.

    Returns:
        Unit quaternion [q0, q1, q2, q3].
    """
    r = np.asarray(dcm, dtype=np.float64)
    trace = np.trace(r)

    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q0 = 0.25 / s
        q1 = (r[1, 2] - r[2, 1]) * s
        q2 = (r[2, 0] - r[0, 2]) * s
        q3 = (r[0, 1] - r[1, 0]) * s
    else:
        if r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
            q0 = (r[1, 2] - r[2, 1]) / s
            q1 = 0.25 * s
            q2 = (r[0, 1] + r[1, 0]) / s
            q3 = (r[0, 2] + r[2, 0]) / s
        elif r[1, 1] > r[2, 2]:
            s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
            q0 = (r[2, 0] - r[0, 2]) / s
            q1 = (r[0, 1] + r[1, 0]) / s
            q2 = 0.25 * s
            q3 = (r[1, 2] + r[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
            q0 = (r[0, 1] - r[1, 0]) / s
            q1 = (r[0, 2] + r[2, 0]) / s
            q2 = (r[1, 2] + r[2, 1]) / s
            q3 = 0.25 * s

    return quat_normalize(np.array([q0, q1, q2, q3], dtype=np.float64))


def omega_matrix(omega_b: np.ndarray) -> np.ndarray:
    """Skew-symmetric quaternion kinematic matrix Omega(omega_B).

    Implements the exact matrix from Section 4.2.3:
    Omega(omega_B) = [
        [0,        -omega_x, -omega_y, -omega_z],
        [omega_x,   0,        omega_z, -omega_y],
        [omega_y,  -omega_z,  0,        omega_x],
        [omega_z,   omega_y, -omega_x,  0      ]
    ]

    Args:
        omega_b: Body angular velocity vector [wx, wy, wz] (rad/s).

    Returns:
        4x4 kinematic transition matrix.
    """
    wx, wy, wz = omega_b[0], omega_b[1], omega_b[2]
    return np.array(
        [
            [0.0, -wx, -wy, -wz],
            [wx, 0.0, wz, -wy],
            [wy, -wz, 0.0, wx],
            [wz, wy, -wx, 0.0],
        ],
        dtype=np.float64,
    )


def quat_derivative(q: np.ndarray, omega_b: np.ndarray) -> np.ndarray:
    """Compute time derivative of attitude quaternion: q_dot = 0.5 * Omega(omega_b) @ q.

    Args:
        q: Attitude quaternion [q0, q1, q2, q3].
        omega_b: Angular rate vector [wx, wy, wz] in body frame (rad/s).

    Returns:
        Time derivative dq/dt of shape (4,).
    """
    omega_mat = omega_matrix(omega_b)
    return 0.5 * (omega_mat @ q)


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles (roll=phi, pitch=theta, yaw=psi in radians, Z-Y-X sequence)
    to scalar-first quaternion q_I^B.
    """
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    q0 = cr * cp * cy + sr * sp * sy
    q1 = sr * cp * cy - cr * sp * sy
    q2 = cr * sp * cy + sr * cp * sy
    q3 = cr * cp * sy - sr * sp * cy

    return quat_normalize(np.array([q0, q1, q2, q3], dtype=np.float64))


def quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    """Convert scalar-first quaternion q_I^B to Euler angles (roll, pitch, yaw) in radians.
    Sequence: Z-Y-X.
    """
    q0, q1, q2, q3 = quat_normalize(q)

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (q0 * q2 - q3 * q1)
    if np.abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def rotate_vector_i_to_b(q: np.ndarray, v_i: np.ndarray) -> np.ndarray:
    """Rotate 3D vector from Inertial to Body frame: v_B = R_I^B @ v_I."""
    return quat_to_dcm_ib(q) @ v_i


def rotate_vector_b_to_i(q: np.ndarray, v_b: np.ndarray) -> np.ndarray:
    """Rotate 3D vector from Body to Inertial frame: v_I = R_B^I @ v_B."""
    return quat_to_dcm_bi(q) @ v_b

