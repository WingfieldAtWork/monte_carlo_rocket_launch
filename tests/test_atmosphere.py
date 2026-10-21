"""Tabular verification of US Standard Atmosphere 1976 model."""

import numpy as np
import pytest

from core.atmosphere import us_standard_atmosphere, geometric_to_geopotential


def test_sea_level_values():
    """Verify standard sea level values (0 m geometric altitude)."""
    atm = us_standard_atmosphere(0.0)
    assert np.isclose(atm.temperature, 288.15, atol=1e-2)
    assert np.isclose(atm.pressure, 101325.0, atol=1.0)
    assert np.isclose(atm.density, 1.2250, atol=1e-3)
    assert np.isclose(atm.speed_of_sound, 340.294, atol=0.2)


def test_tabular_standard_altitudes():
    """Verify temperature and pressure against published US Standard Atmosphere 1976 tables."""
    # Geopotential Altitude [m], Expected Temp [K] (+/-1K), Expected Pressure [Pa] (+/-1%)
    tabular_checkpoints = [
        (11000.0, 216.65, 22632.06),
        (20000.0, 216.65, 5474.89),
        (32000.0, 228.65, 868.019),
        (47000.0, 270.65, 110.906),
        (51000.0, 270.65, 66.939),
        (71000.0, 214.65, 3.956),
    ]

    for alt_h, t_expected, p_expected in tabular_checkpoints:
        atm = us_standard_atmosphere(alt_h, is_geopotential=True)
        assert np.isclose(atm.temperature, t_expected, atol=1.0), f"Failed T at {alt_h}m"
        rel_err_p = abs(atm.pressure - p_expected) / p_expected
        assert rel_err_p < 0.01, f"Failed P at {alt_h}m: got {atm.pressure}, expected {p_expected}"


def test_vacuum_transition():
    """Verify that density and pressure decay strictly to 0 above 86 km."""
    atm_86km = us_standard_atmosphere(86000.0)
    assert atm_86km.density == 0.0
    assert atm_86km.pressure == 0.0

    atm_120km = us_standard_atmosphere(120000.0)
    assert atm_120km.density == 0.0
    assert atm_120km.pressure == 0.0
    assert atm_120km.speed_of_sound > 0.0


def test_monotonic_pressure_decrease():
    """Verify pressure decreases strictly monotonically from 0 to 86 km."""
    altitudes = np.linspace(0, 85000, 100)
    pressures = [us_standard_atmosphere(h).pressure for h in altitudes]

    for i in range(len(pressures) - 1):
        assert pressures[i] > pressures[i + 1]


def test_speed_of_sound_relation():
    """Verify ideal gas relation a = sqrt(gamma * R * T)."""
    for alt in [0.0, 5000.0, 15000.0, 30000.0, 60000.0]:
        atm = us_standard_atmosphere(alt)
        expected_a = np.sqrt(1.4 * 287.05287 * atm.temperature)
        assert np.isclose(atm.speed_of_sound, expected_a, atol=1e-4)
