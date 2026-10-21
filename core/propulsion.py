"""Propulsion and TVC actuation module: thrust, mass depletion, variable CG, and inertia."""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

# Standard gravitational acceleration [m/s^2]
G0: float = 9.80665


@dataclass
class PropulsionConfig:
    """Propulsion specifications."""
    thrust_vac: float         # Vacuum thrust [N]
    isp_vac: float            # Vacuum specific impulse [s]
    exit_area: float          # Nozzle exit plane area [m^2]
    gimbal_limit_deg: float   # Maximum gimbal deflection [deg]
    actuator_tau: float       # First-order actuator time constant [s]


@dataclass
class VehicleMassProperties:
    """Vehicle geometric, mass, and inertia properties."""
    dry_mass: float
    propellant_mass_initial: float
    cg_dry: np.ndarray        # [x, y, z] from base [m]
    cg_wet: np.ndarray
    inertia_dry: np.ndarray   # [Ixx, Iyy, Izz] [kg*m^2]
    inertia_wet: np.ndarray
    r_nozzle: np.ndarray = None  # Nozzle pivot point from base [m]

    def __post_init__(self):
        if self.r_nozzle is None:
            self.r_nozzle = np.array([0.0, 0.0, 0.0], dtype=np.float64)


@dataclass
class PropulsionState:
    """Instantaneous propulsion outputs."""
    thrust_mag: float
    m_dot: float              # Mass flow rate [kg/s] (positive value)
    is_burning: bool
    f_thrust_b: np.ndarray    # Thrust force in Body frame [N]
    m_thrust_b: np.ndarray    # TVC moment in Body frame [N*m]
    delta_y: float            # Current pitch gimbal angle [rad]
    delta_z: float            # Current yaw gimbal angle [rad]
    cg: np.ndarray            # Current CG [m]
    inertia: np.ndarray       # Current principal moments [Ixx, Iyy, Izz] [kg*m^2]


class PropulsionSystem:
    """Manages thrust generation, TVC actuation lag, and mass properties depletion."""

    def __init__(self, prop_config: PropulsionConfig, vehicle_props: VehicleMassProperties):
        self.config = prop_config
        self.props = vehicle_props

        # Actuator states (actual gimbal angles in radians)
        self.delta_y: float = 0.0
        self.delta_z: float = 0.0

        # Precached mass properties
        self.cg_dry_x = float(self.props.cg_dry[0])
        self.cg_wet_x = float(self.props.cg_wet[0])
        self.ixx_dry = float(self.props.inertia_dry[0])
        self.iyy_dry = float(self.props.inertia_dry[1])
        self.izz_dry = float(self.props.inertia_dry[2])
        self.ixx_wet = float(self.props.inertia_wet[0])
        self.iyy_wet = float(self.props.inertia_wet[1])
        self.izz_wet = float(self.props.inertia_wet[2])
        self.dry_mass = float(self.props.dry_mass)
        self.prop_initial = float(self.props.propellant_mass_initial)

        # Nominal mass flow rate dm/dt = T_vac / (Isp_vac * g0)
        self.m_dot_nominal: float = self.config.thrust_vac / (self.config.isp_vac * G0)
        self.gimbal_limit_rad: float = math.radians(self.config.gimbal_limit_deg)

    def get_mass_depletion_rate(self, current_mass: float, t: float, t_burnout: float) -> float:
        """Calculate dm/dt (negative derivative for mass depletion)."""
        if t < t_burnout and (current_mass - self.dry_mass) > 1e-3:
            return -self.m_dot_nominal
        return 0.0

    def get_mass_properties(self, current_mass: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute instantaneous Center of Gravity and Moments of Inertia via linear interpolation."""
        m_prop = max(0.0, current_mass - self.dry_mass)
        f_prop = min(1.0, max(0.0, m_prop / self.prop_initial))

        cg = self.props.cg_dry + f_prop * (self.props.cg_wet - self.props.cg_dry)
        inertia = self.props.inertia_dry + f_prop * (self.props.inertia_wet - self.props.inertia_dry)
        return cg, inertia

    def update_actuators(self, delta_y_cmd: float, delta_z_cmd: float, dt: float) -> None:
        """Update TVC gimbal angles with first-order actuator lag."""
        y_cmd_clamped = min(self.gimbal_limit_rad, max(-self.gimbal_limit_rad, delta_y_cmd))
        z_cmd_clamped = min(self.gimbal_limit_rad, max(-self.gimbal_limit_rad, delta_z_cmd))

        tau = max(1e-4, self.config.actuator_tau)
        alpha = 1.0 - math.exp(-dt / tau)

        self.delta_y += alpha * (y_cmd_clamped - self.delta_y)
        self.delta_z += alpha * (z_cmd_clamped - self.delta_z)

        self.delta_y = min(self.gimbal_limit_rad, max(-self.gimbal_limit_rad, self.delta_y))
        self.delta_z = min(self.gimbal_limit_rad, max(-self.gimbal_limit_rad, self.delta_z))

    def evaluate_fast(
        self,
        current_mass: float,
        ambient_pressure: float,
        t: float,
        t_burnout: float,
    ) -> tuple[float, float, float, float, float, float, float, bool, float, float, float, float]:
        """Fast scalar evaluation of propulsion state.

        Returns:
            (f_tx, f_ty, f_tz, m_tx, m_ty, m_tz, thrust, is_burning, cg_x, ixx, iyy, izz)
        """
        m_prop = max(0.0, current_mass - self.dry_mass)
        f_prop = min(1.0, max(0.0, m_prop / self.prop_initial))

        cg_x = self.cg_dry_x + f_prop * (self.cg_wet_x - self.cg_dry_x)
        ixx = self.ixx_dry + f_prop * (self.ixx_wet - self.ixx_dry)
        iyy = self.iyy_dry + f_prop * (self.iyy_wet - self.iyy_dry)
        izz = self.izz_dry + f_prop * (self.izz_wet - self.izz_dry)

        is_burning = (t < t_burnout) and (m_prop > 1e-3)
        if is_burning:
            thrust = max(0.0, self.config.thrust_vac - self.config.exit_area * ambient_pressure)
        else:
            thrust = 0.0

        cy = math.cos(self.delta_y)
        cz = math.cos(self.delta_z)
        sy = math.sin(self.delta_y)
        sz = math.sin(self.delta_z)

        # F_thrust,B = T * [cos(delta_y)*cos(delta_z), sin(delta_z), -sin(delta_y)]
        f_tx = thrust * (cy * cz)
        f_ty = thrust * sz
        f_tz = -thrust * sy

        # TVC moment M = (r_nozzle - r_CG) x F_thrust
        # r_nozzle = [0, 0, 0], r_CG = [cg_x, 0, 0] -> [-cg_x, 0, 0] x [fx, fy, fz]
        # Mx = 0, My = cg_x * fz = -cg_x * T * sin(delta_y), Mz = -cg_x * fy = -cg_x * T * sin(delta_z)
        m_tx = 0.0
        m_ty = cg_x * f_tz
        m_tz = -cg_x * f_ty

        return f_tx, f_ty, f_tz, m_tx, m_ty, m_tz, thrust, is_burning, cg_x, ixx, iyy, izz

    def evaluate(
        self,
        current_mass: float,
        ambient_pressure: float,
        t: float,
        t_burnout: float,
    ) -> PropulsionState:
        """Full dataclass wrapper for compatibility."""
        f_tx, f_ty, f_tz, m_tx, m_ty, m_tz, thrust, is_burning, cg_x, ixx, iyy, izz = self.evaluate_fast(
            current_mass, ambient_pressure, t, t_burnout
        )
        return PropulsionState(
            thrust_mag=thrust,
            m_dot=self.m_dot_nominal if is_burning else 0.0,
            is_burning=is_burning,
            f_thrust_b=np.array([f_tx, f_ty, f_tz], dtype=np.float64),
            m_thrust_b=np.array([m_tx, m_ty, m_tz], dtype=np.float64),
            delta_y=self.delta_y,
            delta_z=self.delta_z,
            cg=np.array([cg_x, 0.0, 0.0], dtype=np.float64),
            inertia=np.array([ixx, iyy, izz], dtype=np.float64),
        )

