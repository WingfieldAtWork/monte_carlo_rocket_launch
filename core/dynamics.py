"""14-state RK4 flight dynamics integrator for 6-DoF launch vehicle ascent."""

from __future__ import annotations
from typing import Callable
import math
import numpy as np

from core.coordinates import quat_normalize

# State vector indices
POS_IDX = slice(0, 3)
VEL_IDX = slice(3, 6)
QUAT_IDX = slice(6, 10)
OMEGA_IDX = slice(10, 13)
MASS_IDX = 13


def state_derivatives_fast(
    t: float,
    state: np.ndarray,
    forces_and_moments_fn: Callable,
    out_deriv: np.ndarray,
) -> None:
    """Compute state derivatives dx/dt directly into out_deriv without heap allocations."""
    # Slices
    rx = state[0]
    ry = state[1]
    rz = state[2]
    vx = state[3]
    vy = state[4]
    vz = state[5]
    q0 = state[6]
    q1 = state[7]
    q2 = state[8]
    q3 = state[9]
    wx = state[10]
    wy = state[11]
    wz = state[12]
    mass = max(1.0, float(state[13]))

    # Call evaluator returning forces, moments, gravity, inertia, m_dot
    res = forces_and_moments_fn(t, state)
    if len(res) == 13:
        f_tx, f_ty, f_tz, m_tx, m_ty, m_tz, gx, gy, gz, ixx, iyy, izz, m_dot = res
    else:
        fb, mb, gi, inertia, m_dot = res
        f_tx, f_ty, f_tz = float(fb[0]), float(fb[1]), float(fb[2])
        m_tx, m_ty, m_tz = float(mb[0]), float(mb[1]), float(mb[2])
        gx, gy, gz = float(gi[0]), float(gi[1]), float(gi[2])
        ixx, iyy, izz = float(inertia[0]), float(inertia[1]), float(inertia[2])
        m_dot = float(m_dot)

    # 1. dr/dt = v
    out_deriv[0] = vx
    out_deriv[1] = vy
    out_deriv[2] = vz

    # 2. dv/dt = (1/m) * R_B^I * F_B + g_I
    # Rotate F_B from Body to Inertial:
    q0_2 = q0 * q0
    q1_2 = q1 * q1
    q2_2 = q2 * q2
    q3_2 = q3 * q3

    f_ix = (q0_2 + q1_2 - q2_2 - q3_2) * f_tx + 2.0 * (q1 * q2 - q0 * q3) * f_ty + 2.0 * (q1 * q3 + q0 * q2) * f_tz
    f_iy = 2.0 * (q1 * q2 + q0 * q3) * f_tx + (q0_2 - q1_2 + q2_2 - q3_2) * f_ty + 2.0 * (q2 * q3 - q0 * q1) * f_tz
    f_iz = 2.0 * (q1 * q3 - q0 * q2) * f_tx + 2.0 * (q2 * q3 + q0 * q1) * f_ty + (q0_2 - q1_2 - q2_2 + q3_2) * f_tz

    inv_m = 1.0 / mass
    out_deriv[3] = inv_m * f_ix + gx
    out_deriv[4] = inv_m * f_iy + gy
    out_deriv[5] = inv_m * f_iz + gz

    # 3. dq/dt = 0.5 * Omega(omega_b) * q
    out_deriv[6] = 0.5 * (-wx * q1 - wy * q2 - wz * q3)
    out_deriv[7] = 0.5 * (wx * q0 + wz * q2 - wy * q3)
    out_deriv[8] = 0.5 * (wy * q0 - wz * q1 + wx * q3)
    out_deriv[9] = 0.5 * (wz * q0 + wy * q1 - wx * q2)

    # 4. domega/dt = I^-1 * (M - omega x (I * omega))
    # Gyroscopic torque:
    # (omega x (I * omega))_x = wy * (izz * wz) - wz * (iyy * wy) = (izz - iyy) * wy * wz
    # (omega x (I * omega))_y = wz * (ixx * wx) - wx * (izz * wz) = (ixx - izz) * wz * wx
    # (omega x (I * omega))_z = wx * (iyy * wy) - wy * (ixx * wx) = (iyy - ixx) * wx * wy
    out_deriv[10] = (m_tx - (izz - iyy) * wy * wz) / ixx
    out_deriv[11] = (m_ty - (ixx - izz) * wz * wx) / iyy
    out_deriv[12] = (m_tz - (iyy - ixx) * wx * wy) / izz

    # 5. dm/dt
    out_deriv[13] = m_dot


def rk4_step(
    t: float,
    state: np.ndarray,
    dt: float,
    forces_and_moments_fn: Callable,
    m_dry: float,
) -> np.ndarray:
    """Execute a single Runge-Kutta 4th Order step with optimized array buffers."""
    k1 = np.empty(14, dtype=np.float64)
    k2 = np.empty(14, dtype=np.float64)
    k3 = np.empty(14, dtype=np.float64)
    k4 = np.empty(14, dtype=np.float64)
    st_temp = np.empty(14, dtype=np.float64)

    half_dt = 0.5 * dt

    # k1
    state_derivatives_fast(t, state, forces_and_moments_fn, k1)

    # k2
    st_temp[:] = state + half_dt * k1
    state_derivatives_fast(t + half_dt, st_temp, forces_and_moments_fn, k2)

    # k3
    st_temp[:] = state + half_dt * k2
    state_derivatives_fast(t + half_dt, st_temp, forces_and_moments_fn, k3)

    # k4
    st_temp[:] = state + dt * k3
    state_derivatives_fast(t + dt, st_temp, forces_and_moments_fn, k4)

    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Unit quaternion normalization:
    q0 = next_state[6]
    q1 = next_state[7]
    q2 = next_state[8]
    q3 = next_state[9]
    norm = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
    if norm > 1e-12:
        inv_n = 1.0 / norm
        next_state[6] = q0 * inv_n
        next_state[7] = q1 * inv_n
        next_state[8] = q2 * inv_n
        next_state[9] = q3 * inv_n
    else:
        next_state[6] = 1.0
        next_state[7] = 0.0
        next_state[8] = 0.0
        next_state[9] = 0.0

    if next_state[13] < m_dry:
        next_state[13] = m_dry

    return next_state


def state_derivatives(t: float, state: np.ndarray, forces_and_moments_fn: Callable) -> np.ndarray:
    """Wrapper function for compatibility with test_dynamics.py."""
    dstate = np.empty(14, dtype=np.float64)

    def compat_eval(tt, st):
        fb, mb, gi, inertia, m_dot = forces_and_moments_fn(tt, st)
        return (
            float(fb[0]), float(fb[1]), float(fb[2]),
            float(mb[0]), float(mb[1]), float(mb[2]),
            float(gi[0]), float(gi[1]), float(gi[2]),
            float(inertia[0]), float(inertia[1]), float(inertia[2]),
            float(m_dot),
        )

    state_derivatives_fast(t, state, compat_eval, dstate)
    return dstate

