"""Unit tests for RK4 integrator: energy conservation, convergence order, and Euler dynamics."""

import numpy as np
import pytest

from core.coordinates import quat_normalize, quat_to_dcm_bi
from core.dynamics import rk4_step, state_derivatives, POS_IDX, VEL_IDX, QUAT_IDX, OMEGA_IDX, MASS_IDX
from sim.environment import MU_EARTH, R_EARTH


def test_keplerian_energy_and_momentum_conservation():
    """Verify specific mechanical energy and angular momentum are conserved in unperturbed orbit."""
    # Orbit at altitude 200 km
    r0 = R_EARTH + 200000.0
    v0 = np.sqrt(MU_EARTH / r0)  # Circular orbital speed

    # Initial state: in inertial frame, pos = [0, 0, 200000], Earth center at [0, 0, -R_EARTH]
    state = np.zeros(14, dtype=np.float64)
    state[POS_IDX] = np.array([r0, 0.0, -R_EARTH])  # r_EC = [r0, 0, 0]
    state[VEL_IDX] = np.array([0.0, v0, 0.0])
    state[QUAT_IDX] = np.array([1.0, 0.0, 0.0, 0.0])
    state[MASS_IDX] = 12000.0

    # No thrust, no aero, pure point-mass gravity
    def orbital_forces_fn(t, st):
        pos_i = st[POS_IDX]
        r_ec = np.array([pos_i[0], pos_i[1], pos_i[2] + R_EARTH])
        r_norm = float(np.linalg.norm(r_ec))
        g_i = -(MU_EARTH / (r_norm ** 3)) * r_ec
        f_b = np.zeros(3)
        m_b = np.zeros(3)
        inertia = np.array([25000.0, 1200000.0, 1200000.0])
        return f_b, m_b, g_i, inertia, 0.0

    # Initial energy & momentum
    r_ec_init = np.array([state[0], state[1], state[2] + R_EARTH])
    v_init = state[VEL_IDX]
    e0 = 0.5 * np.dot(v_init, v_init) - MU_EARTH / np.linalg.norm(r_ec_init)
    h0 = np.cross(r_ec_init, v_init)

    # Integrate 200 seconds with dt = 0.1s
    dt = 0.1
    t = 0.0
    for _ in range(2000):
        state = rk4_step(t, state, dt, orbital_forces_fn, m_dry=12000.0)
        t += dt

    r_ec_final = np.array([state[0], state[1], state[2] + R_EARTH])
    v_final = state[VEL_IDX]
    e_final = 0.5 * np.dot(v_final, v_final) - MU_EARTH / np.linalg.norm(r_ec_final)
    h_final = np.cross(r_ec_final, v_final)

    # Relative energy error should be under 1e-7
    rel_energy_err = abs((e_final - e0) / e0)
    assert rel_energy_err < 1e-7, f"Energy drift too high: {rel_energy_err}"

    # Angular momentum vector conservation
    assert np.allclose(h0, h_final, rtol=1e-6)


def test_rk4_convergence_rate():
    """Verify 4th-order convergence of the RK4 step through step-halving test."""
    # Test on harmonic oscillator / circular motion
    omega = 0.1  # rad/s

    def linear_dynamics(t, st):
        f_b = np.zeros(3)
        m_b = np.zeros(3)
        # g_i = -omega^2 * r
        g_i = -(omega ** 2) * st[POS_IDX]
        inertia = np.array([100.0, 100.0, 100.0])
        return f_b, m_b, g_i, inertia, 0.0

    state_init = np.zeros(14)
    state_init[POS_IDX] = np.array([10.0, 0.0, 0.0])
    state_init[VEL_IDX] = np.array([0.0, 10.0 * omega, 0.0])
    state_init[QUAT_IDX] = np.array([1.0, 0.0, 0.0, 0.0])
    state_init[MASS_IDX] = 100.0

    t_final = 10.0

    # Run with dt1 = 0.2 s
    dt1 = 0.2
    s1 = state_init.copy()
    t = 0.0
    while t < t_final - 1e-9:
        s1 = rk4_step(t, s1, dt1, linear_dynamics, m_dry=100.0)
        t += dt1

    # Run with dt2 = 0.1 s
    dt2 = 0.1
    s2 = state_init.copy()
    t = 0.0
    while t < t_final - 1e-9:
        s2 = rk4_step(t, s2, dt2, linear_dynamics, m_dry=100.0)
        t += dt2

    # Run with reference fine step dt_ref = 0.01 s
    dt_ref = 0.01
    s_ref = state_init.copy()
    t = 0.0
    while t < t_final - 1e-9:
        s_ref = rk4_step(t, s_ref, dt_ref, linear_dynamics, m_dry=100.0)
        t += dt_ref

    err1 = np.linalg.norm(s1[POS_IDX] - s_ref[POS_IDX])
    err2 = np.linalg.norm(s2[POS_IDX] - s_ref[POS_IDX])

    # Halving the step size in RK4 should reduce error by ~2^4 = 16
    reduction_ratio = err1 / err2
    assert 12.0 < reduction_ratio < 18.0, f"Expected 4th order ratio ~16, got {reduction_ratio}"


def test_quaternion_normalization_guaranteed():
    """Verify that after every RK4 step, quaternion magnitude is identically 1."""
    state = np.zeros(14)
    state[QUAT_IDX] = quat_normalize(np.array([1.0, 0.5, -0.2, 0.3]))
    state[OMEGA_IDX] = np.array([0.2, -0.5, 0.4])  # Fast tumbling
    state[MASS_IDX] = 1000.0

    def tumbling_fn(t, st):
        return np.zeros(3), np.zeros(3), np.zeros(3), np.array([1000.0, 2000.0, 2000.0]), 0.0

    for step in range(100):
        state = rk4_step(step * 0.01, state, 0.01, tumbling_fn, m_dry=500.0)
        q_norm = float(np.linalg.norm(state[QUAT_IDX]))
        assert abs(q_norm - 1.0) < 1e-12

