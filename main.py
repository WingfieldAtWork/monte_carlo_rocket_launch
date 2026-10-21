"""Main CLI entrypoint for 6-DoF launch vehicle ascent and Monte Carlo dispersion engine."""

from __future__ import annotations
import argparse
import sys
import os

from sim.simulation import RocketSimulation, SimulationConfig
from sim.monte_carlo import MonteCarloEngine
from analysis.plotting import plot_nominal_trajectory, plot_monte_carlo_results


def run_nominal(args: argparse.Namespace) -> int:
    """Execute single nominal flight."""
    print("=" * 70)
    print("6-DoF LAUNCH VEHICLE ASCENT SIMULATION (NOMINAL FLIGHT)")
    print("=" * 70)
    print(f"Vehicle Config: {args.config}")
    print(f"Stream UDP:     {args.stream_udp} ({args.udp_host}:{args.udp_port})")

    sim_cfg = SimulationConfig(
        dt_sim=0.005,
        dt_gnc=0.02,
        dt_telem=0.01,
        dt_log=0.05,
        t_max=250.0,
        stream_udp=args.stream_udp,
        udp_host=args.udp_host,
        udp_port=args.udp_port,
    )

    sim = RocketSimulation(vehicle_config_path=args.config, sim_config=sim_cfg)
    summary = sim.run()

    print("\n" + "-" * 40 + " FLIGHT SUMMARY " + "-" * 40)
    print(f"Termination Reason:         {summary.termination_reason}")
    print(f"Burnout Time (MECO):        {summary.burnout_time_s:.2f} s")
    print(f"Insertion Velocity:         {summary.insertion_velocity_mps:.2f} m/s")
    print(f"Orbital Apogee Altitude:    {summary.apogee_altitude_m / 1000.0:.2f} km")
    print(f"Terminal Downrange:         {summary.terminal_downrange_m / 1000.0:.2f} km")
    print(f"Maximum Dynamic Pressure:   {summary.max_q_pa / 1000.0:.2f} kPa (at t = {summary.t_max_q_s:.2f} s)")
    print(f"Max Angle of Attack (q>10k): {summary.max_alpha_at_high_q_deg:.3f} deg")
    print(f"Max Quaternion Norm Error:  {summary.max_quat_norm_error:.2e}")
    print(f"Residual Fuel Margin:       {summary.fuel_margin_kg:.2f} kg")
    print("-" * 96)

    # Always save a copy of diagnostic plot to results/
    plot_save_path = "results/nominal_flight.png"
    if args.plot:
        plot_nominal_trajectory(sim.history, save_path=plot_save_path, show=True)
    else:
        # Save plot silently without popping up window
        plot_nominal_trajectory(sim.history, save_path=plot_save_path, show=False)

    return 0


def run_monte_carlo(args: argparse.Namespace) -> int:
    """Execute multi-core Monte Carlo dispersion analysis."""
    print("=" * 70)
    print("MONTE CARLO DISPERSION ANALYSIS")
    print("=" * 70)
    print(f"Vehicle Config:     {args.config}")
    print(f"Monte Carlo Config: {args.mc_config}")
    print(f"Total Runs:         {args.runs}")
    print(f"Worker Processes:   {args.workers or 'Auto (all CPU cores)'}")
    print(f"Output File:        {args.output or 'results/mc_results.csv'}")

    out_csv = args.output or "results/mc_results.csv"

    engine = MonteCarloEngine(
        vehicle_cfg_path=args.config,
        mc_config_path=args.mc_config,
        num_runs=args.runs,
        max_workers=args.workers,
    )

    results = engine.execute(output_csv=out_csv)

    # Compute key statistics
    import numpy as np

    apogees = [r["apogee_alt_km"] for r in results]
    downranges = [r["terminal_downrange_km"] for r in results]
    max_qs = [r["max_q_kpa"] for r in results]
    v_inss = [r["insertion_velocity_mps"] for r in results]

    print("\n" + "-" * 35 + " DISPERSION STATISTICS " + "-" * 35)
    print(f"Apogee Altitude (km):     Mean={np.mean(apogees):.1f}, Std={np.std(apogees):.1f}, Min={np.min(apogees):.1f}, Max={np.max(apogees):.1f}")
    print(f"Terminal Downrange (km):  Mean={np.mean(downranges):.1f}, Std={np.std(downranges):.1f}, Min={np.min(downranges):.1f}, Max={np.max(downranges):.1f}")
    print(f"Max Dynamic Pressure(kPa):Mean={np.mean(max_qs):.1f}, Std={np.std(max_qs):.1f}, Min={np.min(max_qs):.1f}, Max={np.max(max_qs):.1f}")
    print(f"Insertion Velocity (m/s): Mean={np.mean(v_inss):.1f}, Std={np.std(v_inss):.1f}, Min={np.min(v_inss):.1f}, Max={np.max(v_inss):.1f}")
    print("-" * 93)

    mc_plot_path = "results/mc_analysis.png"
    if args.plot:
        plot_monte_carlo_results(results, save_path=mc_plot_path, show=True)
    else:
        plot_monte_carlo_results(results, save_path=mc_plot_path, show=False)

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="6-DoF Launch Vehicle Ascent & Monte Carlo Simulation Engine"
    )

    parser.add_argument(
        "--mode",
        choices=["nominal", "monte-carlo"],
        default="nominal",
        help="Execution mode: 'nominal' single run or 'monte-carlo' batch dispersion",
    )
    parser.add_argument(
        "--config",
        default="config/vehicle_default.yaml",
        help="Path to vehicle YAML configuration file (default: config/vehicle_default.yaml)",
    )
    parser.add_argument(
        "--mc-config",
        default="config/monte_carlo_config.yaml",
        help="Path to Monte Carlo YAML configuration file (default: config/monte_carlo_config.yaml)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display interactive diagnostic plots upon completion",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=200,
        help="Number of Monte Carlo runs (default: 200)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: all available CPU cores)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write Monte Carlo results CSV (default: results/mc_results.csv)",
    )
    parser.add_argument(
        "--stream-udp",
        action="store_true",
        help="Stream 100 Hz binary telemetry over UDP socket",
    )
    parser.add_argument(
        "--udp-host",
        type=str,
        default="127.0.0.1",
        help="UDP broadcast destination IP address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--udp-port",
        type=int,
        default=9099,
        help="UDP broadcast destination port (default: 9099)",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "nominal":
        return run_nominal(args)
    elif args.mode == "monte-carlo":
        return run_monte_carlo(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

