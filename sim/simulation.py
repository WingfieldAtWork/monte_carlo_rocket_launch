"""Single-run master orchestrator and multi-rate execution loop."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import math
import yaml
import numpy as np

from core.coordinates import quat_to_euler, quat_to_dcm_ib
from core.atmosphere import get_atmosphere_fast, us_standard_atmosphere
from core.aerodynamics import compute_aerodynamics_fast, compute_aerodynamics, AerodynamicState
from core.propulsion import PropulsionSystem, PropulsionConfig, VehicleMassProperties, PropulsionState
from core.dynamics import rk4_step, POS_IDX, VEL_IDX, QUAT_IDX, OMEGA_IDX, MASS_IDX
from gnc.guidance import GuidanceSystem, GuidanceConfig, FlightPhase
from gnc.control import AttitudeController, ControlConfig
from sim.environment import Environment, WindConfig, MU_EARTH, R_EARTH
from sim.hitl_telemetry import HITLTelemetryBroadcaster


@dataclass
class SimulationConfig:
    """Master simulation configuration."""
    dt_sim: float = 0.005         # 200 Hz dynamics RK4 step
    dt_gnc: float = 0.02          # 50 Hz GNC guidance & control loop
    dt_telem: float = 0.01        # 100 Hz HITL UDP telemetry
    dt_log: float = 0.05          # 20 Hz data logging
    t_max: float = 250.0          # Maximum simulation duration [s]
    stream_udp: bool = False
    udp_host: str = "127.0.0.1"
    udp_port: int = 9099


@dataclass
class FlightSummary:
    """End of flight summary metrics."""
    apogee_altitude_m: float
    terminal_downrange_m: float
    max_q_pa: float
    t_max_q_s: float
    insertion_velocity_mps: float
    fuel_margin_kg: float
    burnout_time_s: float
    max_alpha_at_high_q_deg: float
    max_quat_norm_error: float
    termination_reason: str


class RocketSimulation:
    """Master multi-rate 6-DoF launch vehicle ascent simulation orchestrator."""

    def __init__(
        self,
        vehicle_config_path: str,
        sim_config: SimulationConfig | None = None,
        dispersions: dict[str, Any] | None = None,
    ):
        self.sim_cfg = sim_config or SimulationConfig()
        self.dispersions = dispersions or {}

        # Load vehicle YAML config
        with open(vehicle_config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        v_cfg = cfg["vehicle"]
        p_cfg = cfg["propulsion"]
        g_cfg = cfg.get("gnc", {})

        # Apply dispersions if provided
        thrust_mult = 1.0 + self.dispersions.get("thrust_vac_pct", 0.0) / 100.0
        isp_mult = 1.0 + self.dispersions.get("isp_vac_pct", 0.0) / 100.0
        fuel_mult = 1.0 + self.dispersions.get("fuel_mass_pct", 0.0) / 100.0
        self.cd_mult = 1.0 + self.dispersions.get("drag_cd_pct", 0.0) / 100.0

        prop_mass = float(v_cfg["propellant_mass"]) * fuel_mult
        dry_mass = float(v_cfg["dry_mass"])
        self.initial_total_mass = dry_mass + prop_mass
        self.dry_mass = dry_mass

        thrust_vac = float(p_cfg["thrust_vac"]) * thrust_mult
        isp_vac = float(p_cfg["isp_vac"]) * isp_mult

        self.prop_system = PropulsionSystem(
            prop_config=PropulsionConfig(
                thrust_vac=thrust_vac,
                isp_vac=isp_vac,
                exit_area=float(p_cfg["exit_area"]),
                gimbal_limit_deg=float(p_cfg["gimbal_limit_deg"]),
                actuator_tau=float(p_cfg["actuator_tau"]),
            ),
            vehicle_props=VehicleMassProperties(
                dry_mass=dry_mass,
                propellant_mass_initial=prop_mass,
                cg_dry=np.array(v_cfg["cg_dry"], dtype=np.float64),
                cg_wet=np.array(v_cfg["cg_wet"], dtype=np.float64),
                inertia_dry=np.array(v_cfg["inertia_dry"], dtype=np.float64),
                inertia_wet=np.array(v_cfg["inertia_wet"], dtype=np.float64),
            ),
        )

        self.s_ref = float(v_cfg["s_ref"])
        self.t_burnout = prop_mass / self.prop_system.m_dot_nominal

        # Guidance & Control
        self.guidance = GuidanceSystem(
            GuidanceConfig(
                pitch_kick_start_s=float(g_cfg.get("pitch_kick_start_s", 5.0)),
                pitch_kick_duration_s=float(g_cfg.get("pitch_kick_duration_s", 7.0)),
                pitch_kick_rate_deg_s=float(g_cfg.get("pitch_kick_rate_deg_s", 0.8)),
            )
        )
        self.controller = AttitudeController(
            ControlConfig(
                control_kp=float(g_cfg.get("control_kp", 8.0)),
                control_kd=float(g_cfg.get("control_kd", 3.5)),
                gimbal_limit_deg=float(p_cfg["gimbal_limit_deg"]),
            )
        )

        # Environment
        wind_gust_max = float(self.dispersions.get("wind_gust_max_mps", 0.0))
        self.env = Environment(
            WindConfig(
                max_gust_speed=wind_gust_max,
                altitude_min=float(self.dispersions.get("wind_altitude_min_m", 6000.0)),
                altitude_max=float(self.dispersions.get("wind_altitude_max_m", 16000.0)),
                azimuth_deg=float(self.dispersions.get("wind_azimuth_deg", 90.0)),
                gust_seed=self.dispersions.get("wind_seed", None),
            )
        )

        # Telemetry
        self.telemetry = HITLTelemetryBroadcaster(
            host=self.sim_cfg.udp_host,
            port=self.sim_cfg.udp_port,
            enabled=self.sim_cfg.stream_udp,
        )

        # Initial state [rx, ry, rz, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, m]
        # At launch pad: pos = [0, 0, 0], vel = [0, 0, 0]
        # Vertical attitude: nose (+x_B) along +z_I -> q_I^B = [1/sqrt(2), 0, -1/sqrt(2), 0]
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        q_init = np.array([inv_sqrt2, 0.0, -inv_sqrt2, 0.0], dtype=np.float64)

        self.state = np.zeros(14, dtype=np.float64)
        self.state[QUAT_IDX] = q_init
        self.state[MASS_IDX] = self.initial_total_mass

        # Cached GNC commands
        self.delta_y_cmd: float = 0.0
        self.delta_z_cmd: float = 0.0
        self.q_target: np.ndarray = q_init.copy()
        self.current_phase: FlightPhase = FlightPhase.VERTICAL_ASCENT

        # Logging storage
        self.history: dict[str, list] = {
            "time": [],
            "altitude": [],
            "downrange": [],
            "pos_x": [],
            "pos_y": [],
            "pos_z": [],
            "vel_x": [],
            "vel_y": [],
            "vel_z": [],
            "vel_mag": [],
            "mach": [],
            "dyn_pressure": [],
            "alpha_deg": [],
            "beta_deg": [],
            "thrust": [],
            "mass": [],
            "pitch_deg": [],
            "yaw_deg": [],
            "roll_deg": [],
            "rates_p": [],
            "rates_q": [],
            "rates_r": [],
            "gimbal_y_deg": [],
            "gimbal_z_deg": [],
            "phase": [],
            "quat_norm_err": [],
        }

    def _eval_fast_derivatives(self, t: float, current_state: np.ndarray):
        """Fast scalar subsystem callback for dynamics derivative evaluation."""
        rx = current_state[0]
        ry = current_state[1]
        rz = current_state[2]
        vx = current_state[3]
        vy = current_state[4]
        vz = current_state[5]
        q0 = current_state[6]
        q1 = current_state[7]
        q2 = current_state[8]
        q3 = current_state[9]
        mass = current_state[13]

        # 1. Environment & Gravity
        rz_ec = rz + R_EARTH
        r_sq = rx * rx + ry * ry + rz_ec * rz_ec
        r_norm = math.sqrt(r_sq)
        alt = r_norm - R_EARTH

        inv_r3 = -MU_EARTH / (r_norm * r_sq)
        gx = inv_r3 * rx
        gy = inv_r3 * ry
        gz = inv_r3 * rz_ec

        # 2. Atmosphere
        density, pressure, temp, a_sound = get_atmosphere_fast(alt)

        # 3. Wind
        if self.env.wind_cfg.max_gust_speed > 1e-4 and (self.env.wind_cfg.altitude_min <= alt <= self.env.wind_cfg.altitude_max):
            norm_alt = (alt - self.env.peak_altitude) / self.env.shear_width
            speed_fraction = math.exp(-0.5 * (norm_alt * norm_alt))
            gust_noise = 0.15 * math.sin(2.0 * math.pi * 0.2 * t) + 0.1 * math.cos(2.0 * math.pi * 0.05 * t)
            total_speed = self.env.wind_cfg.max_gust_speed * speed_fraction * (1.0 + gust_noise)
            az_rad = math.radians(self.env.wind_cfg.azimuth_deg)
            wx_wind = total_speed * math.cos(az_rad)
            wy_wind = total_speed * math.sin(az_rad)
        else:
            wx_wind = 0.0
            wy_wind = 0.0

        # Relative velocity in inertial frame
        vx_rel = vx - wx_wind
        vy_rel = vy - wy_wind
        vz_rel = vz

        # 4. Propulsion
        f_tx, f_ty, f_tz, m_tx, m_ty, m_tz, thrust, is_burning, cg_x, ixx, iyy, izz = self.prop_system.evaluate_fast(
            mass, pressure, t, self.t_burnout
        )
        m_dot_neg = self.prop_system.get_mass_depletion_rate(mass, t, self.t_burnout)

        # 5. Aerodynamics
        f_ax, f_ay, f_az, m_ax, m_ay, m_az, q_inf, mach, alpha, beta, v_mag = compute_aerodynamics_fast(
            vx_rel, vy_rel, vz_rel, q0, q1, q2, q3, density, a_sound, self.s_ref, cg_x, self.cd_mult
        )

        # Total forces and moments in body frame
        f_bx = f_tx + f_ax
        f_by = f_ty + f_ay
        f_bz = f_tz + f_az

        m_bx = m_tx + m_ax
        m_by = m_ty + m_ay
        m_bz = m_tz + m_az

        return f_bx, f_by, f_bz, m_bx, m_by, m_bz, gx, gy, gz, ixx, iyy, izz, m_dot_neg

    def run(self) -> FlightSummary:
        """Execute the multi-rate simulation loop until termination."""
        dt_sim = self.sim_cfg.dt_sim
        dt_gnc = self.sim_cfg.dt_gnc
        dt_telem = self.sim_cfg.dt_telem
        dt_log = self.sim_cfg.dt_log

        t = 0.0
        step = 0

        next_gnc_t = 0.0
        next_telem_t = 0.0
        next_log_t = 0.0

        max_q = 0.0
        t_max_q = 0.0
        max_alpha_high_q = 0.0
        max_quat_norm_err = 0.0
        termination_reason = "TIME_LIMIT"

        while t <= self.sim_cfg.t_max:
            # Check GNC loop execution (50 Hz)
            if t >= next_gnc_t - 1e-9:
                pos_i = self.state[POS_IDX]
                vel_i = self.state[VEL_IDX]
                q_att = self.state[QUAT_IDX]
                omega_b = self.state[OMEGA_IDX]
                mass = self.state[MASS_IDX]

                v_wind_i = self.env.get_wind(pos_i, t)
                v_rel_i = vel_i - v_wind_i
                is_burning = (t < self.t_burnout) and ((mass - self.dry_mass) > 1e-3)

                # Guidance update
                self.q_target, self.current_phase = self.guidance.update(
                    t=t,
                    pos_i=pos_i,
                    v_rel_i=v_rel_i,
                    is_burning=is_burning,
                )

                # Control update
                ctrl_out = self.controller.compute_control(
                    q_current=q_att,
                    q_target=self.q_target,
                    omega_b=omega_b,
                )
                self.delta_y_cmd = ctrl_out.delta_y_cmd
                self.delta_z_cmd = ctrl_out.delta_z_cmd

                next_gnc_t += dt_gnc

            # Actuator lag update (200 Hz)
            self.prop_system.update_actuators(self.delta_y_cmd, self.delta_z_cmd, dt_sim)

            # Check Telemetry broadcast (100 Hz) or Data Logging (20 Hz)
            need_telem = self.sim_cfg.stream_udp and (t >= next_telem_t - 1e-9)
            need_log = (t >= next_log_t - 1e-9)

            if need_telem or need_log:
                pos_i = self.state[POS_IDX]
                vel_i = self.state[VEL_IDX]
                q_att = self.state[QUAT_IDX]
                omega_b = self.state[OMEGA_IDX]
                mass = self.state[MASS_IDX]

                alt = self.env.get_altitude(pos_i)
                downrange = self.env.get_downrange(pos_i)
                atm = us_standard_atmosphere(alt)
                v_wind_i = self.env.get_wind(pos_i, t)

                prop_state = self.prop_system.evaluate(mass, atm.pressure, t, self.t_burnout)
                aero_state = compute_aerodynamics(
                    vel_i, v_wind_i, q_att, atm.density, atm.speed_of_sound, self.s_ref, prop_state.cg[0], self.cd_mult
                )

                q_norm = float(np.linalg.norm(q_att))
                norm_err = abs(q_norm - 1.0)
                if norm_err > max_quat_norm_err:
                    max_quat_norm_err = norm_err

                if aero_state.q_inf > max_q:
                    max_q = aero_state.q_inf
                    t_max_q = t

                if aero_state.q_inf > 10000.0:
                    alpha_deg = abs(math.degrees(aero_state.alpha))
                    if alpha_deg > max_alpha_high_q:
                        max_alpha_high_q = alpha_deg

                if need_telem:
                    self.telemetry.broadcast(
                        t=t,
                        pos_i=pos_i,
                        vel_i=vel_i,
                        quat=q_att,
                        rates_b=omega_b,
                        mass=mass,
                        dyn_pressure=aero_state.q_inf,
                    )
                    next_telem_t += dt_telem

                if need_log:
                    roll_rad, pitch_rad, yaw_rad = quat_to_euler(q_att)
                    self.history["time"].append(t)
                    self.history["altitude"].append(alt)
                    self.history["downrange"].append(downrange)
                    self.history["pos_x"].append(pos_i[0])
                    self.history["pos_y"].append(pos_i[1])
                    self.history["pos_z"].append(pos_i[2])
                    self.history["vel_x"].append(vel_i[0])
                    self.history["vel_y"].append(vel_i[1])
                    self.history["vel_z"].append(vel_i[2])
                    self.history["vel_mag"].append(float(np.linalg.norm(vel_i)))
                    self.history["mach"].append(aero_state.mach)
                    self.history["dyn_pressure"].append(aero_state.q_inf)
                    self.history["alpha_deg"].append(math.degrees(aero_state.alpha))
                    self.history["beta_deg"].append(math.degrees(aero_state.beta))
                    self.history["thrust"].append(prop_state.thrust_mag)
                    self.history["mass"].append(mass)
                    self.history["pitch_deg"].append(math.degrees(pitch_rad))
                    self.history["yaw_deg"].append(math.degrees(yaw_rad))
                    self.history["roll_deg"].append(math.degrees(roll_rad))
                    self.history["rates_p"].append(omega_b[0])
                    self.history["rates_q"].append(omega_b[1])
                    self.history["rates_r"].append(omega_b[2])
                    self.history["gimbal_y_deg"].append(math.degrees(prop_state.delta_y))
                    self.history["gimbal_z_deg"].append(math.degrees(prop_state.delta_z))
                    self.history["phase"].append(self.current_phase.value)
                    self.history["quat_norm_err"].append(norm_err)

                    next_log_t += dt_log

            # Check Termination conditions
            # 1. Ground impact after liftoff
            alt_now = self.env.get_altitude(self.state[POS_IDX])
            if step > 200 and alt_now < 0.0:
                termination_reason = "GROUND_IMPACT"
                break

            # 2. Propellant exhausted & MECO reached
            if t >= self.t_burnout or (self.state[MASS_IDX] - self.dry_mass) < 1.0:
                termination_reason = "MECO_BURNOUT"
                break

            # Execute RK4 step (200 Hz) using fast scalar derivative callback
            self.state = rk4_step(
                t=t,
                state=self.state,
                dt=dt_sim,
                forces_and_moments_fn=self._eval_fast_derivatives,
                m_dry=self.dry_mass,
            )

            t += dt_sim
            step += 1

        self.telemetry.close()

        # Compute final orbital apogee from state at termination
        pos_i = self.state[POS_IDX]
        vel_i = self.state[VEL_IDX]
        mass = self.state[MASS_IDX]

        r_ec = self.env.get_earth_centered_pos(pos_i)
        r_mag = float(np.linalg.norm(r_ec))
        v_mag = float(np.linalg.norm(vel_i))

        # Specific orbital energy epsilon = v^2/2 - mu/r
        eps = 0.5 * (v_mag ** 2) - MU_EARTH / r_mag
        h_vec = np.cross(r_ec, vel_i)
        h_mag = float(np.linalg.norm(h_vec))

        # Eccentricity
        e_term = 1.0 + (2.0 * eps * (h_mag ** 2)) / (MU_EARTH ** 2)
        ecc = math.sqrt(max(0.0, e_term))

        if eps < 0.0:
            a_semi = -MU_EARTH / (2.0 * eps)
            r_apogee = a_semi * (1.0 + ecc)
            apogee_alt = r_apogee - R_EARTH
        else:
            apogee_alt = 999999.0  # Hyperbolic escape

        fuel_margin = max(0.0, mass - self.dry_mass)

        return FlightSummary(
            apogee_altitude_m=float(apogee_alt),
            terminal_downrange_m=self.env.get_downrange(pos_i),
            max_q_pa=float(max_q),
            t_max_q_s=float(t_max_q),
            insertion_velocity_mps=float(v_mag),
            fuel_margin_kg=float(fuel_margin),
            burnout_time_s=float(min(t, self.t_burnout)),
            max_alpha_at_high_q_deg=float(max_alpha_high_q),
            max_quat_norm_error=float(max_quat_norm_err),
            termination_reason=termination_reason,
        )

