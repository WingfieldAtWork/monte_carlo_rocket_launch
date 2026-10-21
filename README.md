# 6-DoF Launch Vehicle Ascent & Monte Carlo Simulation Engine

A modular, high-fidelity Six-Degree-of-Freedom (6-DoF) launch vehicle ascent simulator in Python. The simulator models coupled rigid-body translational and rotational dynamics, variable mass/inertia depletion, US Standard Atmosphere aerothermodynamics, and Thrust Vector Control (TVC) pitch-over guidance. It includes a multi-core Monte Carlo dispersion engine and a 100 Hz binary UDP telemetry streaming module for Hardware-in-the-Loop (HITL) test interfaces.

---

## Features

- **Coupled 6-DoF Flight Dynamics:** 14-state continuous system integrated via Runge-Kutta 4th Order (RK4) with guaranteed unit quaternion normalization ($|\|\mathbf{q}\| - 1| < 10^{-6}$).
- **US Standard Atmosphere 1976:** Accurate atmospheric properties (density, pressure, temperature, speed of sound) across Troposphere, Stratosphere, and Mesosphere with smooth vacuum cutoff above 86 km.
- **Mach-Dependent Aerodynamics:** Transonic drag rise ($C_D(M)$), dynamic pressure ($q$), angles of attack/sideslip ($\alpha, \beta$), aerodynamic center of pressure migration ($CP(M)$), and restoring weathercock stability moments.
- **Propulsion & TVC Gimbal:** Pressure-corrected thrust $T(h) = T_{\text{vac}} - A_e P(h)$, first-order actuator lag ($\tau = 0.05\text{ s}$), linear mass/inertia depletion, and TVC gimbal steering moments.
- **GNC Flight Logic:** Three-phase ascent sequencer (vertical rise, constant pitch-over kick, gravity turn velocity tracking) paired with a quaternion-error PD attitude controller.
- **Multi-Core Monte Carlo Engine:** Parallel batch execution using Python's `ProcessPoolExecutor` with Gaussian/uniform dispersions on vacuum thrust, $I_{sp}$, drag, fuel mass loading, and high-altitude wind gusts.
- **HITL Binary Telemetry:** Real-time 100 Hz UDP binary streaming formatted via big-endian struct `>d3f3f4f3fff` (68 bytes).
- **Diagnostic Plotting:** Trajectory profiles, Max-Q aerodynamic loading, attitude and gimbal tracking, and Monte Carlo dispersion envelopes.

---

## Directory Structure

```text
├── config/
│   ├── vehicle_default.yaml       # Mass, inertia, aero reference, TVC limits
│   └── monte_carlo_config.yaml   # Dispersions (thrust, Isp, drag, winds)
├── core/
│   ├── __init__.py
│   ├── coordinates.py             # Quaternion math, DCMs, frame transformations
│   ├── atmosphere.py              # US Standard Atmosphere 1976 model
│   ├── aerodynamics.py            # Mach-dependent Cd, dynamic pressure, aero loads
│   ├── propulsion.py              # Thrust, mass-depletion, variable CG and inertia
│   └── dynamics.py                # 14-state RK4 integrator (F=ma, Euler's equations)
├── gnc/
│   ├── __init__.py
│   ├── guidance.py                # Vertical rise, pitch kick, gravity turn schedule
│   └── control.py                 # Quaternion-error PD controller for TVC gimbal
├── sim/
│   ├── __init__.py
│   ├── environment.py             # Wind shear models, gravity (WGS-84 point mass)
│   ├── simulation.py              # Single-run master orchestrator and multi-rate clock
│   ├── monte_carlo.py             # Multiprocessing dispersion harness
│   └── hitl_telemetry.py          # 100 Hz UDP binary telemetry packet broadcaster
├── analysis/
│   ├── __init__.py
│   └── plotting.py                # Trajectory, Max-Q, attitude error, MC envelopes
├── tests/
│   ├── test_quaternions.py        # Normalization, identity, composition checks
│   ├── test_atmosphere.py         # Tabular checks against US 1976 standard
│   ├── test_dynamics.py           # Energy conservation, RK4 convergence
│   └── test_end_to_end.py         # Full nominal run to MECO validation
├── main.py                        # CLI entrypoint
├── requirements.txt
└── README.md
```

---

## Mathematical & Physics Specifications

### State Vector ($\mathbf{x} \in \mathbb{R}^{14}$)

$$\mathbf{x} = \left[ x, y, z, v_x, v_y, v_z, q_0, q_1, q_2, q_3, \omega_x, \omega_y, \omega_z, m \right]^T$$

- Position $\mathbf{r}_I = [x, y, z]^T$: Launch-centered Inertial Frame ($+z$ Up, $+x$ Downrange, $+y$ Crossrange).
- Velocity $\mathbf{v}_I = [v_x, v_y, v_z]^T$: Inertial translational velocity.
- Attitude Quaternion $\mathbf{q}_I^B = [q_0, q_1, q_2, q_3]^T$: Scalar-first unit quaternion ($||\mathbf{q}|| = 1$).
- Body Angular Velocity $\boldsymbol{\omega}_B = [p, q, r]^T$: Body frame angular rates.
- Instantaneous Mass $m$: Rocket mass (dry mass + propellant).

### Equations of Motion

1. **Translational Kinematics:** $\dot{\mathbf{r}}_I = \mathbf{v}_I$
2. **Translational Dynamics:** $\dot{\mathbf{v}}_I = \frac{1}{m} \mathbf{R}_B^I (\mathbf{F}_{\text{thrust}, B} + \mathbf{F}_{\text{aero}, B}) + \mathbf{g}_I(\mathbf{r})$
3. **Rotational Kinematics:** $\dot{\mathbf{q}} = \frac{1}{2} \boldsymbol{\Omega}(\boldsymbol{\omega}_B) \mathbf{q}$
4. **Rotational Dynamics:** $\dot{\boldsymbol{\omega}}_B = \mathbf{I}^{-1} \left( \mathbf{M}_{\text{total}, B} - \boldsymbol{\omega}_B \times (\mathbf{I} \boldsymbol{\omega}_B) \right)$
5. **Mass Depletion:** $\dot{m} = -\frac{T_{\text{vac}}}{I_{sp, \text{vac}} g_0}$ (active until burnout)

---

## CLI & Execution Commands

### 1. Single Nominal Flight with Diagnostic Plots
```bash
python main.py --mode nominal --config config/vehicle_default.yaml --plot
```

### 2. Monte Carlo Dispersion Analysis (e.g. 200 runs across 8 workers)
```bash
python main.py --mode monte-carlo --runs 200 --workers 8 --output results/mc_run_01.csv
```

### 3. Nominal Flight with HITL UDP Telemetry Streaming (100 Hz)
```bash
python main.py --mode nominal --stream-udp --udp-port 9099
```

### 4. Execute Test Suite
```bash
pytest tests/ -v
```
