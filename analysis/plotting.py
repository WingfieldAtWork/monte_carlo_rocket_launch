"""Diagnostic plotting tools: trajectory profiles, Max-Q aerothermal loading,
attitude errors, and Monte Carlo dispersion envelopes.
"""

from __future__ import annotations
import os
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

# Professional styling
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def plot_nominal_trajectory(
    history: dict[str, list],
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Generate high-fidelity multi-panel diagnostic plots for nominal ascent flight."""
    t = np.array(history["time"])
    alt_km = np.array(history["altitude"]) / 1000.0
    downrange_km = np.array(history["downrange"]) / 1000.0
    vel_mps = np.array(history["vel_mag"])
    mach = np.array(history["mach"])
    q_kpa = np.array(history["dyn_pressure"]) / 1000.0
    alpha_deg = np.array(history["alpha_deg"])
    beta_deg = np.array(history["beta_deg"])
    pitch_deg = np.array(history["pitch_deg"])
    rates_q = np.degrees(np.array(history["rates_q"]))
    gimbal_y = np.array(history["gimbal_y_deg"])
    thrust_kn = np.array(history["thrust"]) / 1000.0

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("6-DoF Launch Vehicle Nominal Ascent Flight Diagnostics", fontsize=16, fontweight="bold")

    # 1. 2D Ascent Profile (Altitude vs Downrange)
    ax1 = axes[0, 0]
    ax1.plot(downrange_km, alt_km, color="royalblue", lw=2.5, label="Ascent Trajectory")
    ax1.scatter([downrange_km[-1]], [alt_km[-1]], color="crimson", s=60, zorder=5, label=f"MECO ({alt_km[-1]:.1f} km)")
    ax1.set_title("Ascent Trajectory (Altitude vs Downrange)", fontweight="bold")
    ax1.set_xlabel("Ground Downrange (km)")
    ax1.set_ylabel("Altitude (km)")
    ax1.legend(loc="upper left")

    # 2. Kinematics (Altitude & Velocity vs Time)
    ax2 = axes[0, 1]
    ax2.plot(t, alt_km, color="forestgreen", lw=2, label="Altitude (km)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Altitude (km)", color="forestgreen")
    ax2.tick_params(axis="y", labelcolor="forestgreen")

    ax2_twin = ax2.twinx()
    ax2_twin.plot(t, vel_mps, color="darkorange", lw=2, linestyle="--", label="Inertial Speed (m/s)")
    ax2_twin.set_ylabel("Speed (m/s)", color="darkorange")
    ax2_twin.tick_params(axis="y", labelcolor="darkorange")
    ax2.set_title("Kinematics Profile", fontweight="bold")

    # 3. Dynamic Pressure & Mach Number (Max-Q)
    ax3 = axes[1, 0]
    max_q_idx = int(np.argmax(q_kpa))
    ax3.plot(t, q_kpa, color="crimson", lw=2, label="Dynamic Pressure q (kPa)")
    ax3.scatter([t[max_q_idx]], [q_kpa[max_q_idx]], color="black", s=70, zorder=5,
                label=f"Max-Q: {q_kpa[max_q_idx]:.1f} kPa @ {t[max_q_idx]:.1f}s")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Dynamic Pressure (kPa)", color="crimson")
    ax3.tick_params(axis="y", labelcolor="crimson")

    ax3_twin = ax3.twinx()
    ax3_twin.plot(t, mach, color="purple", lw=1.8, linestyle=":", label="Mach Number")
    ax3_twin.set_ylabel("Mach Number", color="purple")
    ax3_twin.tick_params(axis="y", labelcolor="purple")
    ax3.set_title("Aerodynamic Loading & Max-Q", fontweight="bold")
    ax3.legend(loc="upper left")

    # 4. Aerodynamic Angles of Attack & Sideslip
    ax4 = axes[1, 1]
    ax4.plot(t, alpha_deg, color="navy", lw=1.8, label="Angle of Attack (alpha)")
    ax4.plot(t, beta_deg, color="darkcyan", lw=1.8, linestyle="--", label="Sideslip (beta)")
    ax4.axhline(2.0, color="red", linestyle=":", alpha=0.7, label="+/- 2.0 deg limit")
    ax4.axhline(-2.0, color="red", linestyle=":", alpha=0.7)
    ax4.set_title("Aerodynamic Flow Angles", fontweight="bold")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Angle (deg)")
    ax4.set_ylim(-3.0, 3.0)
    ax4.legend(loc="upper right")

    # 5. Pitch Attitude & Pitch Rate
    ax5 = axes[2, 0]
    ax5.plot(t, pitch_deg, color="firebrick", lw=2, label="Pitch Angle (deg)")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Pitch Angle (deg)", color="firebrick")
    ax5.tick_params(axis="y", labelcolor="firebrick")

    ax5_twin = ax5.twinx()
    ax5_twin.plot(t, rates_q, color="teal", lw=1.5, linestyle="--", label="Pitch Rate q (deg/s)")
    ax5_twin.set_ylabel("Pitch Rate (deg/s)", color="teal")
    ax5_twin.tick_params(axis="y", labelcolor="teal")
    ax5.set_title("Pitch Attitude & Rate Evolution", fontweight="bold")

    # 6. TVC Gimbal Actuation & Thrust
    ax6 = axes[2, 1]
    ax6.plot(t, gimbal_y, color="mediumvioletred", lw=1.8, label="TVC Pitch Deflection (delta_y)")
    ax6.axhline(6.0, color="gray", linestyle=":", alpha=0.6, label="Gimbal Limit (+/- 6 deg)")
    ax6.axhline(-6.0, color="gray", linestyle=":", alpha=0.6)
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Gimbal Angle (deg)", color="mediumvioletred")
    ax6.tick_params(axis="y", labelcolor="mediumvioletred")

    ax6_twin = ax6.twinx()
    ax6_twin.plot(t, thrust_kn, color="black", lw=1.5, linestyle="-.", label="Thrust (kN)")
    ax6_twin.set_ylabel("Thrust (kN)", color="black")
    ax6_twin.tick_params(axis="y", labelcolor="black")
    ax6.set_title("TVC Gimbal Actuation & Propulsion", fontweight="bold")
    ax6.legend(loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"[Plotting] Nominal flight plot saved to: {save_path}")

    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


def plot_monte_carlo_results(
    results: list[dict[str, Any]],
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Generate diagnostic statistical distribution and dispersion envelope plots for Monte Carlo results."""
    if not results:
        print("[Plotting Warning] No Monte Carlo results to plot.")
        return

    apogees = np.array([r["apogee_alt_km"] for r in results])
    downranges = np.array([r["terminal_downrange_km"] for r in results])
    max_qs = np.array([r["max_q_kpa"] for r in results])
    v_inss = np.array([r["insertion_velocity_mps"] for r in results])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Monte Carlo Dispersion Analysis ({len(results)} Runs)", fontsize=16, fontweight="bold")

    # 1. Apogee Altitude Distribution
    ax1 = axes[0, 0]
    mu_a, std_a = np.mean(apogees), np.std(apogees)
    ax1.hist(apogees, bins=25, color="cornflowerblue", edgecolor="black", alpha=0.7)
    ax1.axvline(mu_a, color="red", lw=2, label=f"Mean: {mu_a:.1f} km")
    ax1.axvline(mu_a - 3 * std_a, color="darkred", linestyle="--", lw=1.5, label=f"-3 sigma: {mu_a - 3*std_a:.1f} km")
    ax1.axvline(mu_a + 3 * std_a, color="darkred", linestyle="--", lw=1.5, label=f"+3 sigma: {mu_a + 3*std_a:.1f} km")
    ax1.axvline(160.0, color="green", linestyle=":", lw=2, label="Spec Minimum (160 km)")
    ax1.set_title("Apogee Altitude Distribution", fontweight="bold")
    ax1.set_xlabel("Apogee Altitude (km)")
    ax1.set_ylabel("Count")
    ax1.legend(loc="upper right")

    # 2. Max Dynamic Pressure (Max-Q) Distribution
    ax2 = axes[0, 1]
    mu_q, std_q = np.mean(max_qs), np.std(max_qs)
    ax2.hist(max_qs, bins=25, color="salmon", edgecolor="black", alpha=0.7)
    ax2.axvline(mu_q, color="black", lw=2, label=f"Mean: {mu_q:.1f} kPa")
    ax2.axvline(mu_q - 3 * std_q, color="firebrick", linestyle="--", lw=1.5, label=f"-3 sigma: {mu_q - 3*std_q:.1f} kPa")
    ax2.axvline(mu_q + 3 * std_q, color="firebrick", linestyle="--", lw=1.5, label=f"+3 sigma: {mu_q + 3*std_q:.1f} kPa")
    ax2.set_title("Max-Q Dynamic Pressure Distribution", fontweight="bold")
    ax2.set_xlabel("Max-Q (kPa)")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper right")

    # 3. Downrange vs Apogee Scatter
    ax3 = axes[1, 0]
    sc = ax3.scatter(downranges, apogees, c=max_qs, cmap="viridis", alpha=0.8, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax3)
    cb.set_label("Max-Q (kPa)")
    ax3.set_title("Terminal Downrange vs Apogee Altitude", fontweight="bold")
    ax3.set_xlabel("Terminal Downrange (km)")
    ax3.set_ylabel("Apogee Altitude (km)")

    # 4. Insertion Velocity vs Apogee
    ax4 = axes[1, 1]
    ax4.scatter(v_inss, apogees, color="darkorchid", alpha=0.7, edgecolors="black", s=35)
    ax4.set_title("Insertion Velocity vs Apogee", fontweight="bold")
    ax4.set_xlabel("Burnout Insertion Velocity (m/s)")
    ax4.set_ylabel("Apogee Altitude (km)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"[Plotting] Monte Carlo dispersion plot saved to: {save_path}")

    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)

