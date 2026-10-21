"""US Standard Atmosphere 1976 model up to 86 km with vacuum transition beyond.

Implements standard geopotential atmospheric layers:
- Troposphere (0 to 11 km)
- Stratosphere (11 to 47 km)
- Mesosphere (47 to 86 km)
Returns density, pressure, temperature, and speed of sound.
"""

from __future__ import annotations
from dataclasses import dataclass
import math

# Physical constants for US Standard Atmosphere 1976
G0: float = 9.80665  # Standard gravity at sea level [m/s^2]
R_AIR: float = 287.05287  # Specific gas constant for dry air [J/(kg*K)]
GAMMA: float = 1.4  # Ratio of specific heats for dry air
EARTH_RADIUS_ATM: float = 6356766.0  # Effective Earth radius for geopotential altitude [m]
ATMOSPHERE_CEILING_M: float = 86000.0  # Altitude above which vacuum is assumed [m]

# Base values at each layer boundary in geopotential meters:
# H0 = 0 m: T = 288.15 K, P = 101325.0 Pa, L = -0.0065 K/m
# H1 = 11000 m: T = 216.65 K, P = 22632.06 Pa, L = 0.0 K/m
# H2 = 20000 m: T = 216.65 K, P = 5474.89 Pa, L = +0.0010 K/m
# H3 = 32000 m: T = 228.65 K, P = 868.019 Pa, L = +0.0028 K/m
# H4 = 47000 m: T = 270.65 K, P = 110.906 Pa, L = 0.0 K/m
# H5 = 51000 m: T = 270.65 K, P = 66.939 Pa, L = -0.0028 K/m
# H6 = 71000 m: T = 214.65 K, P = 3.956 Pa, L = -0.0020 K/m
# H7 = 84852 m: upper limit

P0 = 101325.0
T0 = 288.15
L0 = -0.0065
EXP0 = -G0 / (R_AIR * L0)

P1 = P0 * ((T0 + L0 * 11000.0) / T0) ** EXP0
T1 = 216.65

P2 = P1 * math.exp(-G0 * 9000.0 / (R_AIR * T1))
T2 = 216.65
L2 = 0.0010
EXP2 = -G0 / (R_AIR * L2)

P3 = P2 * ((T2 + L2 * 12000.0) / T2) ** EXP2
T3 = 228.65
L3 = 0.0028
EXP3 = -G0 / (R_AIR * L3)

P4 = P3 * ((T3 + L3 * 15000.0) / T3) ** EXP3
T4 = 270.65

P5 = P4 * math.exp(-G0 * 4000.0 / (R_AIR * T4))
T5 = 270.65
L5 = -0.0028
EXP5 = -G0 / (R_AIR * L5)

P6 = P5 * ((T5 + L5 * 20000.0) / T5) ** EXP5
T6 = 214.65
L6 = -0.0020
EXP6 = -G0 / (R_AIR * L6)


@dataclass(frozen=True)
class AtmosphericProperties:
    """Atmospheric state at a given altitude."""
    altitude: float        # Geometric altitude [m]
    density: float         # Air density rho [kg/m^3]
    pressure: float        # Static pressure P [Pa]
    temperature: float     # Static temperature T [K]
    speed_of_sound: float  # Speed of sound a [m/s]


def geometric_to_geopotential(h: float) -> float:
    """Convert geometric altitude h to geopotential altitude H."""
    return (EARTH_RADIUS_ATM * h) / (EARTH_RADIUS_ATM + h)


def get_atmosphere_fast(h_m: float) -> tuple[float, float, float, float]:
    """Fast inline computation of (density, pressure, temperature, speed_of_sound).

    Optimized for high-frequency RK4 integration loop (zero heap allocations).
    """
    if h_m >= ATMOSPHERE_CEILING_M:
        t_exo = 186.87
        a_exo = math.sqrt(GAMMA * R_AIR * t_exo)
        return 0.0, 0.0, t_exo, a_exo

    if h_m <= 0.0:
        a0 = math.sqrt(GAMMA * R_AIR * T0)
        return P0 / (R_AIR * T0), P0, T0, a0

    # Geopotential altitude
    H = (EARTH_RADIUS_ATM * h_m) / (EARTH_RADIUS_ATM + h_m)

    if H < 11000.0:
        T = T0 + L0 * H
        P = P0 * (T / T0) ** EXP0
    elif H < 20000.0:
        T = T1
        P = P1 * math.exp(-G0 * (H - 11000.0) / (R_AIR * T1))
    elif H < 32000.0:
        T = T2 + L2 * (H - 20000.0)
        P = P2 * (T / T2) ** EXP2
    elif H < 47000.0:
        T = T3 + L3 * (H - 32000.0)
        P = P3 * (T / T3) ** EXP3
    elif H < 51000.0:
        T = T4
        P = P4 * math.exp(-G0 * (H - 47000.0) / (R_AIR * T4))
    elif H < 71000.0:
        T = T5 + L5 * (H - 51000.0)
        P = P5 * (T / T5) ** EXP5
    elif H < 84852.0:
        T = T6 + L6 * (H - 71000.0)
        P = P6 * (T / T6) ** EXP6
    else:
        t_exo = 186.87
        a_exo = math.sqrt(GAMMA * R_AIR * t_exo)
        return 0.0, 0.0, t_exo, a_exo

    rho = P / (R_AIR * T)
    a = math.sqrt(GAMMA * R_AIR * T)
    return rho, P, T, a


def us_standard_atmosphere(altitude_m: float, is_geopotential: bool = False) -> AtmosphericProperties:
    """Compute US Standard Atmosphere 1976 properties up to 86 km.

    Args:
        altitude_m: Altitude in meters.
        is_geopotential: If True, altitude_m is treated as geopotential altitude H;
                         if False (default), altitude_m is geometric altitude h.

    Returns:
        AtmosphericProperties containing (altitude, density, pressure, temperature, speed_of_sound).
    """
    h = max(0.0, float(altitude_m))

    if h >= ATMOSPHERE_CEILING_M:
        t_exo = 186.87
        a_exo = math.sqrt(GAMMA * R_AIR * t_exo)
        return AtmosphericProperties(
            altitude=h,
            density=0.0,
            pressure=0.0,
            temperature=t_exo,
            speed_of_sound=a_exo,
        )

    H = h if is_geopotential else geometric_to_geopotential(h)

    if H < 11000.0:
        T = T0 + L0 * H
        P = P0 * (T / T0) ** EXP0
    elif H < 20000.0:
        T = T1
        P = P1 * math.exp(-G0 * (H - 11000.0) / (R_AIR * T1))
    elif H < 32000.0:
        T = T2 + L2 * (H - 20000.0)
        P = P2 * (T / T2) ** EXP2
    elif H < 47000.0:
        T = T3 + L3 * (H - 32000.0)
        P = P3 * (T / T3) ** EXP3
    elif H < 51000.0:
        T = T4
        P = P4 * math.exp(-G0 * (H - 47000.0) / (R_AIR * T4))
    elif H < 71000.0:
        T = T5 + L5 * (H - 51000.0)
        P = P5 * (T / T5) ** EXP5
    elif H < 84852.0:
        T = T6 + L6 * (H - 71000.0)
        P = P6 * (T / T6) ** EXP6
    else:
        t_exo = 186.87
        a_exo = math.sqrt(GAMMA * R_AIR * t_exo)
        return AtmosphericProperties(altitude=h, density=0.0, pressure=0.0, temperature=t_exo, speed_of_sound=a_exo)

    rho = P / (R_AIR * T)
    a = math.sqrt(GAMMA * R_AIR * T)

    return AtmosphericProperties(
        altitude=h,
        density=float(rho),
        pressure=float(P),
        temperature=float(T),
        speed_of_sound=float(a),
    )

