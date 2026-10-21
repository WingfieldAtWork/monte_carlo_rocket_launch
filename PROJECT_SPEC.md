Here is a complete, self-contained Markdown engineering specification designed to be handed directly to an AI coding agent (e.g., Claude Code, Cursor, Codex) to build the repository from scratch.

Save the content below as `PROJECT_SPEC.md` and provide it to the agent with the command:

`"Implement the repository according to the specifications in PROJECT_SPEC.md. Ensure all unit tests pass and sample runs generate diagnostic plots."`

---

```markdown
# Engineering Specification: 6-DoF Launch Vehicle Ascent & Monte Carlo Simulation Engine

## 1. Project Goal & Overview
Build a modular, high-fidelity Six-Degree-of-Freedom (6-DoF) launch vehicle ascent simulator in Python. The simulator models coupled rigid-body translational and rotational dynamics, variable mass/inertia depletion, US Standard Atmosphere aerothermodynamics, and Thrust Vector Control (TVC) pitch-over guidance. It includes a multi-core Monte Carlo dispersion engine and a 100 Hz binary UDP telemetry streaming module for Hardware-in-the-Loop (HITL) test interfaces.

---

## 2. Tech Stack & Dependencies
* **Language:** Python 3.10+
* **Core Numerics:** `numpy`, `scipy`
* **Configuration:** `pyyaml`
* **Plotting & Diagnostics:** `matplotlib`
* **Testing:** `pytest`
* **Concurrency:** Standard library `concurrent.futures` / `multiprocessing`
* **Networking/Serialization:** Standard library `socket`, `struct`

---

## 3. Directory Structure

```text
rocket_6dof/
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

## 4. Mathematical & Physics Specifications

### 4.1 State Vector Definition ($\mathbf{x} \in \mathbb{R}^{14}$)

The state vector consists of 14 continuous states integrated over time:


$$\mathbf{x} = \left[ x, y, z, v_x, v_y, v_z, q_0, q_1, q_2, q_3, \omega_x, \omega_y, \omega_z, m \right]^T$$

* **Position $\mathbf{r}_{I} = [x, y, z]^T$:** Launch-centered Inertial Frame (meters).
* **Velocity $\mathbf{v}_{I} = [v_x, v_y, v_z]^T$:** Inertial translational velocity (m/s).
* **Attitude Quaternion $\mathbf{q}_{I}^B = [q_0, q_1, q_2, q_3]^T$:** Rotation from Inertial to Body frame (scalar-first convention, $\Vert{}\mathbf{q}\Vert{} = 1$).
* **Angular Velocity $\boldsymbol{\omega}_{B} = [\omega_x, \omega_y, \omega_z]^T$:** Body-frame rotation rates $[p, q, r]^T$ (rad/s).
* **Vehicle Mass $m$:** Instantaneous mass (kg).

### 4.2 Equations of Motion (RK4 Derivatives)

1. **Translational Kinematics:**

$$\dot{\mathbf{r}}_{I} = \mathbf{v}_{I}$$


2. **Translational Dynamics:**

$$\dot{\mathbf{v}}_{I} = \frac{1}{m} \mathbf{R}_{B}^I (\mathbf{F}_{\text{thrust}, B} + \mathbf{F}_{\text{aero}, B}) + \mathbf{g}_{I}(\mathbf{r})$$


* Where $\mathbf{R}_B^I$ is the Direction Cosine Matrix derived from $\mathbf{q}$.
* Gravity: $\mathbf{g}_I(\mathbf{r}) = -\frac{\mu}{\Vert{}\mathbf{r}\Vert{}^3} \mathbf{r}$, with $\mu = 3.986004418 \times 10^{14} \text{ m}^3/\text{s}^2$.


3. **Rotational Kinematics (Quaternion Rate):**

$$\dot{\mathbf{q}} = \frac{1}{2} \boldsymbol{\Omega}(\boldsymbol{\omega}_B) \mathbf{q} = \frac{1}{2} \begin{bmatrix} 0 & -\omega_x & -\omega_y & -\omega_z \\ \omega_x & 0 & \omega_z & -\omega_y \\ \omega_y & -\omega_z & 0 & \omega_x \\ \omega_z & \omega_y & -\omega_x & 0 \end{bmatrix} \begin{bmatrix} q_0 \\ q_1 \\ q_2 \\ q_3 \end{bmatrix}$$


* **Constraint:** Enforce $\Vert{}\mathbf{q}\Vert{} = 1$ via normalization after each RK4 full step.


4. **Rotational Dynamics (Euler’s Equations):**

$$\dot{\boldsymbol{\omega}}_B = \mathbf{I}^{-1} \left( \mathbf{M}_{\text{total}, B} - \boldsymbol{\omega}_B \times (\mathbf{I} \boldsymbol{\omega}_B) - \dot{\mathbf{I}} \boldsymbol{\omega}_B \right)$$


* Assume $\dot{\mathbf{I}} \approx 0$ over short integration steps.
* $\mathbf{I} = \text{diag}(I_{xx}, I_{yy}, I_{zz})$ updated dynamically based on remaining propellant.


5. **Mass Depletion:**

$$\dot{m} = \begin{cases} -\frac{T_{\text{vac}}}{I_{sp, \text{vac}} \cdot g_0} & \text{if } t < t_{\text{burnout}} \text{ and } m > m_{\text{dry}} \\ 0 & \text{otherwise} \end{cases}$$


* Standard gravity constant $g_0 = 9.80665 \text{ m/s}^2$.



---

## 5. Subsystem Implementation Requirements

### 5.1 Environment & Atmosphere (`core/atmosphere.py`)

Implement the **US Standard Atmosphere 1976** up to 86 km:

* Troposphere (0–11 km), Stratosphere (11–47 km), Mesosphere (47–86 km).
* Return density $\rho(h)$, pressure $P(h)$, temperature $T(h)$, and speed of sound $a(h) = \sqrt{\gamma R T}$.
* For altitudes $h > 86 \text{ km}$, assume vacuum ($\rho = 0, P = 0$).

### 5.2 Aerodynamics (`core/aerodynamics.py`)

* Relative velocity: $\mathbf{v}_{\text{rel}, I} = \mathbf{v}_I - \mathbf{v}_{\text{wind}, I}$.
* Convert $\mathbf{v}_{\text{rel}}$ into Body frame: $\mathbf{v}_{\text{rel}, B} = \mathbf{R}_I^B \mathbf{v}_{\text{rel}, I}$.
* Total Mach: $M = \Vert{}\mathbf{v}_{\text{rel}}\Vert{} / a(h)$.
* Dynamic Pressure: $q = \frac{1}{2} \rho(h) \Vert{}\mathbf{v}_{\text{rel}}\Vert{}^2$.
* Angle of Attack ($\alpha$) and Sideslip Angle ($\beta$):

$$\alpha = \arctan2(v_{\text{rel}, B, z}, v_{\text{rel}, B, x}), \quad \beta = \arcsin\left(\frac{v_{\text{rel}, B, y}}{\Vert{}\mathbf{v}_{\text{rel}, B}\Vert{}}\right)$$


* Aerodynamic Forces & Moments:
* $C_D(M)$ lookup table (subsonic $\approx 0.3$, transonic peak at $M=1.05 \approx 0.65$, supersonic drop to $\approx 0.35$).
* $F_{\text{drag}} = q \cdot S_{\text{ref}} \cdot C_D(M)$.
* Restoring stability moment: calculate center of pressure $CP(M)$ and center of gravity $CG(m)$, generating restoring aerodynamic moment $\mathbf{M}_{\text{aero}} = (\mathbf{r}_{CP} - \mathbf{r}_{CG}) \times \mathbf{F}_{\text{aero}}$.



### 5.3 Propulsion & TVC Actuation (`core/propulsion.py`)

* Thrust corrected for ambient pressure:

$$T(h) = T_{\text{vac}} - A_e \cdot P(h)$$


* TVC Gimbal angles: Pitch deflection $\delta_y$, Yaw deflection $\delta_z$ (clamped to $\pm 6.0^\circ$).
* Thrust vector in Body frame:

$$\mathbf{F}_{\text{thrust}, B} = T(h) \begin{bmatrix} \cos\delta_y \cos\delta_z \\ \sin\delta_z \\ -\sin\delta_y \end{bmatrix}$$


* TVC Moment: $\mathbf{M}_{\text{thrust}, B} = (\mathbf{r}_{\text{nozzle}} - \mathbf{r}_{CG}) \times \mathbf{F}_{\text{thrust}, B}$.

### 5.4 GNC Flight Logic (`gnc/`)

* **Flight Phase Sequencer:**
* $t \in [0, 5\text{s}]$: Vertical ascent (keep $\mathbf{q}_{\text{target}}$ pointing pure vertical).
* $t \in [5, 12\text{s}]$: Pitch kick (command constant pitch rate $\dot{\theta} \approx -1.0^\circ/\text{s}$).
* $t > 12\text{s}$ to MECO: Gravity turn (command vehicle to align with velocity vector $\mathbf{v}_{\text{rel}, I}$).


* **Attitude Controller (`gnc/control.py`):**
* Compute error quaternion: $\mathbf{q}_{\text{err}} = \mathbf{q}^{-1} \otimes \mathbf{q}_{\text{target}}$.
* Extract rotation error vector: $\mathbf{e} = 2 \cdot \text{sgn}(q_{\text{err}, 0}) \cdot [q_{\text{err}, 1}, q_{\text{err}, 2}, q_{\text{err}, 3}]^T$.
* Controller: $\mathbf{u} = K_p \mathbf{e} - K_d \boldsymbol{\omega}_B$.
* Map control output to gimbal commands $[\delta_y, \delta_z]$ with a first-order actuator lag (time constant $\tau = 0.05\text{ s}$).



---

## 6. Simulation & Infrastructure

### 6.1 Multi-Rate Execution Loop (`sim/simulation.py`)

* Master Clock loop running at $\Delta t_{\text{sim}} = 0.005\text{ s}$ (200 Hz).
* Dynamics RK4 step: 200 Hz.
* GNC Guidance & Control updates: 50 Hz ($\Delta t = 0.02\text{ s}$).
* Data Logging: 10 Hz ($\Delta t = 0.1\text{ s}$).
* Termination conditions: Target MECO apogee reached, propellant exhausted, or ground impact.

### 6.2 Monte Carlo Engine (`sim/monte_carlo.py`)

* Use `concurrent.futures.ProcessPoolExecutor` to run $N$ runs in parallel across all CPU cores.
* Dispersions drawn from Gaussian/uniform distributions via `monte_carlo_config.yaml`:
* Engine Vacuum Thrust: $\pm 2.0\%$ ($3\sigma$).
* Engine Specific Impulse ($I_{sp}$): $\pm 1.5\%$ ($3\sigma$).
* Total Aerodynamic Drag ($C_D$): $\pm 10.0\%$ ($3\sigma$).
* Fuel Mass Loading: $\pm 0.5\%$ ($3\sigma$).
* High-altitude wind gusts: Pink noise or Dryden shear gust model up to $25\text{ m/s}$.


* Collect metrics: Apogee altitude ($r_a$), terminal downrange, Max-Q load, insertion velocity, fuel margin.

### 6.3 HITL Telemetry Streaming (`sim/hitl_telemetry.py`)

Provide a socket module that serializes the vehicle state every 10 ms (100 Hz) into a binary UDP packet:

* **Format:** Big-endian struct `>d3f3f4f3fff` (Time [double], Pos [3 floats], Vel [3 floats], Quat [4 floats], Rates [3 floats], Mass [float], DynPressure [float]).
* Destination: Configurable IP/Port (default `127.0.0.1:9099`).

---

## 7. Configuration Defaults (`config/vehicle_default.yaml`)

```yaml
vehicle:
  name: "TwoStage-Orbital-Class"
  dry_mass: 12000.0                      # kg
  propellant_mass: 150000.0              # kg
  length: 35.0                           # m
  diameter: 3.0                          # m
  s_ref: 7.068                           # m^2 (frontal area)
  cg_dry: [14.0, 0.0, 0.0]               # m from base
  cg_wet: [19.0, 0.0, 0.0]
  inertia_dry: [25000, 1200000, 1200000] # kg*m^2 [Ixx, Iyy, Izz]
  inertia_wet: [85000, 8500000, 8500000]

propulsion:
  thrust_vac: 2200000.0    # N
  isp_vac: 310.0           # s
  exit_area: 1.25          # m^2
  gimbal_limit_deg: 6.0    # deg
  actuator_tau: 0.05       # s

gnc:
  pitch_kick_start_s: 5.0
  pitch_kick_duration_s: 7.0
  pitch_kick_rate_deg_s: 0.8
  control_kp: 8.0
  control_kd: 3.5

```

---

## 8. CLI & Execution Commands

The entrypoint `main.py` must support the following interface:

```bash
# 1. Run single nominal flight and show interactive trajectory plots
python main.py --mode nominal --config config/vehicle_default.yaml --plot

# 2. Run Monte Carlo dispersion analysis with 200 runs across 8 workers
python main.py --mode monte-carlo --runs 200 --workers 8 --output results/mc_run_01.csv

# 3. Run nominal simulation while broadcasting binary HITL UDP packets to localhost
python main.py --mode nominal --stream-udp --udp-port 9099

# 4. Execute test suite
pytest tests/ -v

```

---

## 9. Verification & Acceptance Criteria

1. **Kinematics:** Quaternion normalization holds ($\left| \|\mathbf{q}\| - 1 \right| < 10^{-6}$) throughout the entire ascent trajectory.
2. **Aerodynamics:** Max-Q occurs between $t = 50\text{ s}$ and $t = 85\text{ s}$, with dynamic pressure correctly decaying to 0 Pa as the rocket exits the atmosphere ($>80\text{ km}$).
3. **Flight Trajectory:** The nominal run successfully performs a pitch kick, maintains angle-of-attack $|\alpha| < 2.0^\circ$ during high dynamic pressure ($q > 10\text{ kPa}$), achieves burnout (MECO), and deposits the vehicle in an elliptical exo-atmospheric trajectory (apogee $> 160\text{ km}$).
4. **Performance:** Monte Carlo engine executes 100 runs in under 45 seconds on an 8-core CPU.
5. **Code Style:** Clean typing annotations, modular function signatures, and zero hardcoded physical constants outside configuration files.

```

***