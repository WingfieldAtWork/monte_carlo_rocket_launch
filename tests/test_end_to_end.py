"""End-to-end integration and verification tests against Acceptance Criteria."""

import os
import numpy as np
import pytest

from sim.simulation import RocketSimulation, SimulationConfig
from sim.hitl_telemetry import HITLTelemetryBroadcaster, PACKET_SIZE


def test_nominal_ascent_end_to_end():
    """Execute full nominal ascent to MECO and validate against all Acceptance Criteria."""
    cfg_path = "config/vehicle_default.yaml"
    assert os.path.exists(cfg_path), f"Configuration file not found: {cfg_path}"

    sim_cfg = SimulationConfig(
        dt_sim=0.005,
        dt_gnc=0.02,
        dt_telem=0.01,
        dt_log=0.05,
        t_max=250.0,
        stream_udp=False,
    )

    sim = RocketSimulation(vehicle_config_path=cfg_path, sim_config=sim_cfg)
    summary = sim.run()

    # Criteria 1: Kinematics quaternion normalization holds (| ||q|| - 1 | < 1e-6)
    assert summary.max_quat_norm_error < 1e-6, (
        f"Quaternion normalization exceeded limit: {summary.max_quat_norm_error}"
    )

    # Criteria 2: Aerodynamics - Max-Q occurs between t = 50s and t = 85s
    assert 50.0 <= summary.t_max_q_s <= 85.0, (
        f"Max-Q time {summary.t_max_q_s:.1f}s not in expected range [50, 85]s"
    )

    # Dynamic pressure correctly decays to ~0 Pa as rocket exits atmosphere (> 80 km)
    final_dyn_p = sim.history["dyn_pressure"][-1]
    final_alt = sim.history["altitude"][-1]
    assert final_alt > 80000.0, f"Rocket did not reach exo-atmosphere: {final_alt}m"
    assert final_dyn_p < 1.0, f"Dynamic pressure did not decay to 0 above 80 km: {final_dyn_p} Pa"

    # Criteria 3: Flight Trajectory
    # - Maintains angle-of-attack |alpha| < 2.0 deg during high dynamic pressure (q > 10 kPa)
    assert summary.max_alpha_at_high_q_deg < 2.0, (
        f"Angle of attack exceeded 2.0 deg during Max-Q: {summary.max_alpha_at_high_q_deg:.3f} deg"
    )

    # - Achieves burnout (MECO)
    assert summary.termination_reason == "MECO_BURNOUT"
    assert summary.fuel_margin_kg < 50.0

    # - Deposits vehicle in elliptical exo-atmospheric trajectory (apogee > 160 km)
    apogee_km = summary.apogee_altitude_m / 1000.0
    assert apogee_km > 160.0, f"Apogee altitude {apogee_km:.1f} km did not exceed 160 km"


def test_hitl_telemetry_packet_serialization():
    """Verify 100 Hz binary UDP telemetry pack/unpack match format >d3f3f4f3fff."""
    t_test = 42.125
    pos_test = np.array([1234.5, -567.8, 9876.5], dtype=np.float32)
    vel_test = np.array([345.6, -12.3, 789.0], dtype=np.float32)
    quat_test = np.array([0.7071, 0.0, -0.7071, 0.0], dtype=np.float32)
    rates_test = np.array([0.01, -0.05, 0.02], dtype=np.float32)
    mass_test = 145000.0
    dyn_p_test = 35400.0

    packet = HITLTelemetryBroadcaster.pack(
        t=t_test,
        pos_i=pos_test,
        vel_i=vel_test,
        quat=quat_test,
        rates_b=rates_test,
        mass=mass_test,
        dyn_pressure=dyn_p_test,
    )

    assert len(packet) == PACKET_SIZE == 68

    unpacked = HITLTelemetryBroadcaster.unpack(packet)
    assert np.isclose(unpacked["time"], t_test)
    assert np.allclose(unpacked["pos"], pos_test, atol=1e-2)
    assert np.allclose(unpacked["vel"], vel_test, atol=1e-2)
    assert np.allclose(unpacked["quat"], quat_test, atol=1e-3)
    assert np.allclose(unpacked["rates"], rates_test, atol=1e-3)
    assert np.isclose(unpacked["mass"], mass_test, rtol=1e-4)
    assert np.isclose(unpacked["dyn_pressure"], dyn_p_test, rtol=1e-4)

