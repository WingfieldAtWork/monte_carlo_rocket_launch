"""Unit tests for quaternion mathematics, DCM transformations, and kinematics."""

import numpy as np
import pytest

from core.coordinates import (
    quat_normalize,
    quat_conjugate,
    quat_multiply,
    quat_to_dcm_ib,
    quat_to_dcm_bi,
    dcm_to_quat,
    omega_matrix,
    quat_derivative,
    euler_to_quat,
    quat_to_euler,
    rotate_vector_i_to_b,
    rotate_vector_b_to_i,
)


def test_quaternion_normalization():
    """Verify quaternion normalization to unit length and edge case handling."""
    # Degenerate zero quaternion returns identity
    q_zero = np.zeros(4)
    q_norm_zero = quat_normalize(q_zero)
    assert np.allclose(q_norm_zero, [1.0, 0.0, 0.0, 0.0])

    # Arbitrary non-unit quaternion
    q_rand = np.array([2.5, -1.2, 3.4, 0.8])
    q_unit = quat_normalize(q_rand)
    assert np.isclose(np.linalg.norm(q_unit), 1.0, atol=1e-12)


def test_quaternion_identity_and_conjugate():
    """Verify identity and conjugate properties."""
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    q_test = quat_normalize(np.array([0.7, 0.2, -0.4, 0.5]))

    # Multiplication by identity on left and right
    assert np.allclose(quat_multiply(q_id, q_test), q_test)
    assert np.allclose(quat_multiply(q_test, q_id), q_test)

    # Conjugate product yields identity
    q_conj = quat_conjugate(q_test)
    assert np.allclose(quat_multiply(q_test, q_conj), q_id, atol=1e-12)
    assert np.allclose(quat_multiply(q_conj, q_test), q_id, atol=1e-12)


def test_quaternion_composition_associativity():
    """Verify associativity: (q1 * q2) * q3 == q1 * (q2 * q3)."""
    rng = np.random.default_rng(12345)
    for _ in range(10):
        q1 = quat_normalize(rng.normal(size=4))
        q2 = quat_normalize(rng.normal(size=4))
        q3 = quat_normalize(rng.normal(size=4))

        p1 = quat_multiply(quat_multiply(q1, q2), q3)
        p2 = quat_multiply(q1, quat_multiply(q2, q3))
        assert np.allclose(p1, p2, atol=1e-12)


def test_dcm_orthonormality_and_roundtrip():
    """Verify Direction Cosine Matrix is orthonormal (det=+1, R*R^T = I) and roundtrips to quat."""
    rng = np.random.default_rng(54321)
    for _ in range(20):
        q = quat_normalize(rng.normal(size=4))

        r_ib = quat_to_dcm_ib(q)
        r_bi = quat_to_dcm_bi(q)

        # Transpose relationship
        assert np.allclose(r_ib.T, r_bi, atol=1e-12)

        # Orthonormality
        assert np.allclose(r_ib @ r_bi, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(r_ib), 1.0, atol=1e-12)

        # Roundtrip conversion
        q_recovered = dcm_to_quat(r_ib)
        # Quaternions q and -q represent identical rotations
        dot_product = abs(float(np.dot(q, q_recovered)))
        assert np.isclose(dot_product, 1.0, atol=1e-10)


def test_vector_rotation():
    """Verify vector rotation between Inertial and Body frames."""
    # Rotate 90 degrees around Y axis
    # Inertial: +z -> Body: +x
    # q = [cos(-45), 0, sin(-45), 0] = [1/sqrt(2), 0, -1/sqrt(2), 0]
    inv_s2 = 1.0 / np.sqrt(2.0)
    q_pitch90 = np.array([inv_s2, 0.0, -inv_s2, 0.0])

    v_inertial_z = np.array([0.0, 0.0, 1.0])
    v_body = rotate_vector_i_to_b(q_pitch90, v_inertial_z)
    assert np.allclose(v_body, [1.0, 0.0, 0.0], atol=1e-12)

    v_recovered_i = rotate_vector_b_to_i(q_pitch90, v_body)
    assert np.allclose(v_recovered_i, v_inertial_z, atol=1e-12)


def test_euler_conversions():
    """Verify Euler angle to quaternion and back."""
    test_angles = [
        (0.0, 0.0, 0.0),
        (0.1, -0.3, 0.4),
        (-0.5, 0.8, -1.2),
        (0.0, np.pi / 2.0 - 0.01, 0.0),
    ]
    for roll, pitch, yaw in test_angles:
        q = euler_to_quat(roll, pitch, yaw)
        r_rec, p_rec, y_rec = quat_to_euler(q)
        assert np.isclose(roll, r_rec, atol=1e-6)
        assert np.isclose(pitch, p_rec, atol=1e-6)
        assert np.isclose(yaw, y_rec, atol=1e-6)


def test_kinematics_norm_invariant():
    """Verify that dq/dt is orthogonal to q (i.e. q^T * dq/dt = 0)."""
    rng = np.random.default_rng(999)
    for _ in range(10):
        q = quat_normalize(rng.normal(size=4))
        omega = rng.normal(size=3) * 0.1  # rad/s

        q_dot = quat_derivative(q, omega)
        # Derivative of ||q||^2 = 2 * q^T * q_dot must be 0
        rate_of_norm_sq = float(np.dot(q, q_dot))
        assert np.isclose(rate_of_norm_sq, 0.0, atol=1e-12)

