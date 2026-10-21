"""Aerodynamics model: Mach-dependent drag, dynamic pressure, angles of attack/sideslip,
and aerodynamic loads & restoring stability moments.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from core.coordinates import quat_to_dcm_ib, rotate_vector_i_to_b

# Slender body normal force slope dC_N / dalpha (rad^-1)
CN_ALPHA_DEFAULT: float = 3.0


@dataclass
class AerodynamicState:
    """Calculated aerodynamic quantities."""
    v_rel_i: np.ndarray        # Relative velocity in Inertial frame [m/s]
    v_rel_b: np.ndarray        # Relative velocity in Body frame [m/s]
    v_rel_mag: float          # Relative airspeed magnitude [m/s]
    mach: float               # Mach number
    q_inf: float              # Dynamic pressure [Pa]
    alpha: float              # Angle of attack [rad]
    beta: float               # Sideslip angle [rad]
    cd: float                 # Total drag coefficient
    f_aero_b: np.ndarray      # Aerodynamic force in Body frame [N]
    m_aero_b: np.ndarray      # Aerodynamic moment in Body frame [N*m]
    cp_x: float               # Center of pressure location along x from base [m]


def get_drag_coefficient(mach: float, cd_multiplier: float = 1.0) -> float:
    """Fast linear interpolation of zero-lift drag coefficient C_D(M).

    Subsonic ~0.30, Transonic peak at M=1.05 ~0.65, Supersonic drop to ~0.35.
    """
    m = float(mach)
    if m <= 0.8:
        cd_base = 0.30 + 0.025 * max(0.0, m)
    elif m <= 1.05:
        cd_base = 0.32 + (0.65 - 0.32) * (m - 0.8) / 0.25
    elif m <= 1.5:
        cd_base = 0.65 - (0.65 - 0.48) * (m - 1.05) / 0.45
    elif m <= 2.0:
        cd_base = 0.48 - (0.48 - 0.40) * (m - 1.5) / 0.50
    elif m <= 3.0:
        cd_base = 0.40 - (0.40 - 0.35) * (m - 2.0) / 1.0
    elif m <= 5.0:
        cd_base = 0.35 - (0.35 - 0.32) * (m - 3.0) / 2.0
    else:
        cd_base = max(0.28, 0.32 - 0.004 * (m - 5.0))

    return cd_base * cd_multiplier


def get_center_of_pressure(mach: float) -> float:
    """Compute Center of Pressure (CP) location from base as a function of Mach."""
    m = max(0.0, float(mach))
    if m < 0.8:
        return 12.2
    elif m < 2.0:
        return 12.2 - 0.8 * (m - 0.8) / 1.2
    else:
        return 11.4 - 0.4 * min((m - 2.0) / 3.0, 1.0)


def compute_aerodynamics_fast(
    vx_rel_i: float,
    vy_rel_i: float,
    vz_rel_i: float,
    q0: float,
    q1: float,
    q2: float,
    q3: float,
    density: float,
    speed_of_sound: float,
    s_ref: float,
    cg_x: float,
    cd_mult: float = 1.0,
    cn_alpha: float = CN_ALPHA_DEFAULT,
) -> tuple[float, float, float, float, float, float, float, float, float, float, float]:
    """High-speed scalar evaluation of aerodynamic forces and moments.

    Returns:
        (f_ax_b, f_ay_b, f_az_b, m_ax_b, m_ay_b, m_az_b, q_inf, mach, alpha, beta, v_mag)
    """
    v_sq = vx_rel_i * vx_rel_i + vy_rel_i * vy_rel_i + vz_rel_i * vz_rel_i
    v_mag = math.sqrt(v_sq)

    if v_mag < 1e-4 or density <= 1e-12:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, v_mag

    # Dynamic pressure
    q_inf = 0.5 * density * v_sq

    # Mach number
    mach = (v_mag / speed_of_sound) if speed_of_sound > 1e-3 else 0.0

    # Rotate v_rel into body frame (R_I^B * v_rel_i)
    q0_2 = q0 * q0
    q1_2 = q1 * q1
    q2_2 = q2 * q2
    q3_2 = q3 * q3

    v_b_x = (q0_2 + q1_2 - q2_2 - q3_2) * vx_rel_i + 2.0 * (q1 * q2 + q0 * q3) * vy_rel_i + 2.0 * (q1 * q3 - q0 * q2) * vz_rel_i
    v_b_y = 2.0 * (q1 * q2 - q0 * q3) * vx_rel_i + (q0_2 - q1_2 + q2_2 - q3_2) * vy_rel_i + 2.0 * (q2 * q3 + q0 * q1) * vz_rel_i
    v_b_z = 2.0 * (q1 * q3 + q0 * q2) * vx_rel_i + 2.0 * (q2 * q3 - q0 * q1) * vy_rel_i + (q0_2 - q1_2 - q2_2 + q3_2) * vz_rel_i

    # Flow angles
    alpha = math.atan2(v_b_z, v_b_x)
    sin_beta = min(1.0, max(-1.0, v_b_y / v_mag))
    beta = math.asin(sin_beta)

    cd = get_drag_coefficient(mach, cd_mult)
    cp_x = get_center_of_pressure(mach)

    # Aerodynamic forces in body frame
    # F_x: axial (drag) opposes forward speed
    f_ax = -q_inf * s_ref * cd * math.cos(alpha) * math.cos(beta)
    f_ay = -q_inf * s_ref * cn_alpha * math.sin(beta)
    f_az = -q_inf * s_ref * cn_alpha * math.sin(alpha)

    # Moments: M = (r_CP - r_CG) x F_aero
    # r_CP - r_CG = [cp_x - cg_x, 0, 0]
    dx = cp_x - cg_x
    m_ax = 0.0
    m_ay = -dx * f_az
    m_az = dx * f_ay

    return f_ax, f_ay, f_az, m_ax, m_ay, m_az, q_inf, mach, alpha, beta, v_mag


def compute_aerodynamics(
    v_i: np.ndarray,
    v_wind_i: np.ndarray,
    q_att: np.ndarray,
    density: float,
    speed_of_sound: float,
    s_ref: float,
    cg_x: float,
    cd_multiplier: float = 1.0,
    cn_alpha: float = CN_ALPHA_DEFAULT,
) -> AerodynamicState:
    """Full dataclass wrapper for compatibility and reporting."""
    v_rel_i = v_i - v_wind_i
    f_ax, f_ay, f_az, m_ax, m_ay, m_az, q_inf, mach, alpha, beta, v_mag = compute_aerodynamics_fast(
        float(v_rel_i[0]), float(v_rel_i[1]), float(v_rel_i[2]),
        float(q_att[0]), float(q_att[1]), float(q_att[2]), float(q_att[3]),
        density, speed_of_sound, s_ref, cg_x, cd_multiplier, cn_alpha,
    )
    r_i_b = quat_to_dcm_ib(q_att)
    v_rel_b = r_i_b @ v_rel_i
    cd = get_drag_coefficient(mach, cd_multiplier)
    cp_x = get_center_of_pressure(mach)

    return AerodynamicState(
        v_rel_i=v_rel_i,
        v_rel_b=v_rel_b,
        v_rel_mag=v_mag,
        mach=mach,
        q_inf=q_inf,
        alpha=alpha,
        beta=beta,
        cd=cd,
        f_aero_b=np.array([f_ax, f_ay, f_az], dtype=np.float64),
        m_aero_b=np.array([m_ax, m_ay, m_az], dtype=np.float64),
        cp_x=cp_x,
    )

