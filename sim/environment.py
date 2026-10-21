"""Environmental models: WGS-84 point-mass gravity and high-altitude wind shear models."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# WGS-84 and Gravitational constants
MU_EARTH: float = 3.986004418e14   # Gravitational parameter [m^3/s^2]
R_EARTH: float = 6378137.0         # Earth equatorial radius [m]


@dataclass
class WindConfig:
    """Wind and gust model configuration."""
    max_gust_speed: float = 0.0     # Peak gust speed [m/s]
    altitude_min: float = 6000.0    # Altitude where shear begins [m]
    altitude_max: float = 16000.0   # Altitude where shear ends [m]
    azimuth_deg: float = 90.0       # Wind blowing direction [deg] (90 = +x downrange)
    gust_seed: int | None = None


class Environment:
    """Provides spherical gravity and altitude-dependent wind field."""

    def __init__(self, wind_config: WindConfig | None = None):
        self.wind_cfg = wind_config or WindConfig()
        self.rng = np.random.default_rng(self.wind_cfg.gust_seed)

        # Generate a smooth high-altitude jet shear / Dryden-like gust profile
        self.peak_altitude: float = 0.5 * (self.wind_cfg.altitude_min + self.wind_cfg.altitude_max)
        self.shear_width: float = 0.5 * (self.wind_cfg.altitude_max - self.wind_cfg.altitude_min)

    def get_earth_centered_pos(self, pos_i: np.ndarray) -> np.ndarray:
        """Vector from Earth center to vehicle in Launch-Centered Inertial Frame.

        Launch pad is at (0, 0, 0), with Earth center located at (0, 0, -R_EARTH).
        """
        return np.array([pos_i[0], pos_i[1], pos_i[2] + R_EARTH], dtype=np.float64)

    def get_altitude(self, pos_i: np.ndarray) -> float:
        """Geometric altitude above Earth spherical surface in meters."""
        r_ec = self.get_earth_centered_pos(pos_i)
        return float(np.linalg.norm(r_ec) - R_EARTH)

    def get_downrange(self, pos_i: np.ndarray) -> float:
        """Great-circle ground downrange distance from launch pad in meters."""
        r_ec = self.get_earth_centered_pos(pos_i)
        horizontal_dist = np.sqrt(r_ec[0] ** 2 + r_ec[1] ** 2)
        angle = np.arctan2(horizontal_dist, r_ec[2])
        return float(R_EARTH * angle)

    def get_gravity(self, pos_i: np.ndarray) -> np.ndarray:
        """Calculate point-mass gravitational acceleration vector g_I in inertial frame:
        g_I(r) = - (mu / ||r_EC||^3) * r_EC
        """
        r_ec = self.get_earth_centered_pos(pos_i)
        r_norm = float(np.linalg.norm(r_ec))
        if r_norm < 1e-3:
            return np.array([0.0, 0.0, -9.80665], dtype=np.float64)
        return -(MU_EARTH / (r_norm ** 3)) * r_ec

    def get_wind(self, pos_i: np.ndarray, t: float) -> np.ndarray:
        """Compute wind velocity vector v_wind_I in inertial frame at given position & time."""
        if self.wind_cfg.max_gust_speed <= 1e-4:
            return np.zeros(3, dtype=np.float64)

        alt = self.get_altitude(pos_i)

        # High-altitude jet stream profile: bell-shaped peak around 11 km
        if self.wind_cfg.altitude_min <= alt <= self.wind_cfg.altitude_max:
            norm_alt = (alt - self.peak_altitude) / max(100.0, self.shear_width)
            speed_fraction = np.exp(-0.5 * (norm_alt ** 2))

            # Add high-frequency turbulence / gust variation
            gust_noise = 0.15 * np.sin(2.0 * np.pi * 0.2 * t) + 0.1 * np.cos(2.0 * np.pi * 0.05 * t)
            total_speed = self.wind_cfg.max_gust_speed * speed_fraction * (1.0 + gust_noise)

            az_rad = np.radians(self.wind_cfg.azimuth_deg)
            # Wind in x-y horizontal plane
            w_x = total_speed * np.cos(az_rad)
            w_y = total_speed * np.sin(az_rad)
            w_z = 0.0
            return np.array([w_x, w_y, w_z], dtype=np.float64)

        return np.zeros(3, dtype=np.float64)

