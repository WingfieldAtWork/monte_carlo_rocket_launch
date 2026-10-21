"""Multiprocessing Monte Carlo dispersion harness using concurrent.futures."""

from __future__ import annotations
import os
import csv
import time
from typing import Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml
import numpy as np

from sim.simulation import RocketSimulation, SimulationConfig, FlightSummary


def _single_run_worker(
    run_id: int,
    vehicle_cfg_path: str,
    dispersions: dict[str, float],
    seed: int,
) -> dict[str, Any]:
    """Worker task executed in a worker process for a single Monte Carlo run."""
    dispersions_with_seed = dict(dispersions)
    dispersions_with_seed["wind_seed"] = seed

    # Slightly reduced logging frequency for fast batch performance
    sim_cfg = SimulationConfig(
        dt_sim=0.005,
        dt_gnc=0.02,
        dt_telem=0.1,
        dt_log=0.5,
        stream_udp=False,
    )

    sim = RocketSimulation(
        vehicle_config_path=vehicle_cfg_path,
        sim_config=sim_cfg,
        dispersions=dispersions_with_seed,
    )
    summary = sim.run()

    return {
        "run_id": run_id,
        "thrust_pct": dispersions["thrust_vac_pct"],
        "isp_pct": dispersions["isp_vac_pct"],
        "drag_pct": dispersions["drag_cd_pct"],
        "fuel_pct": dispersions["fuel_mass_pct"],
        "wind_gust_mps": dispersions["wind_gust_max_mps"],
        "apogee_alt_km": summary.apogee_altitude_m / 1000.0,
        "terminal_downrange_km": summary.terminal_downrange_m / 1000.0,
        "max_q_kpa": summary.max_q_pa / 1000.0,
        "t_max_q_s": summary.t_max_q_s,
        "insertion_velocity_mps": summary.insertion_velocity_mps,
        "fuel_margin_kg": summary.fuel_margin_kg,
        "burnout_time_s": summary.burnout_time_s,
        "termination_reason": summary.termination_reason,
    }


class MonteCarloEngine:
    """Manages generation of dispersed input sets and parallel execution."""

    def __init__(
        self,
        vehicle_cfg_path: str,
        mc_config_path: str,
        num_runs: int = 200,
        max_workers: int | None = None,
        random_seed: int = 42,
    ):
        self.vehicle_cfg_path = vehicle_cfg_path
        self.mc_config_path = mc_config_path
        self.num_runs = num_runs
        self.max_workers = max_workers or os.cpu_count() or 4
        self.rng = np.random.default_rng(random_seed)

        # Load MC configuration
        with open(mc_config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        d_cfg = cfg.get("dispersions", {})
        self.thrust_3sigma = float(d_cfg.get("thrust_vac_pct_3sigma", 2.0))
        self.isp_3sigma = float(d_cfg.get("isp_vac_pct_3sigma", 1.5))
        self.drag_3sigma = float(d_cfg.get("drag_cd_pct_3sigma", 10.0))
        self.fuel_3sigma = float(d_cfg.get("fuel_mass_pct_3sigma", 0.5))
        self.wind_max = float(d_cfg.get("wind_gust_max_mps", 25.0))
        self.wind_alt_min = float(d_cfg.get("wind_altitude_min_m", 6000.0))
        self.wind_alt_max = float(d_cfg.get("wind_altitude_max_m", 16000.0))

    def generate_dispersions(self) -> list[dict[str, float]]:
        """Generate N sets of parameter dispersions drawn from Gaussian/Uniform distributions."""
        dispersions_list = []

        # Gaussian 1-sigma = 3-sigma / 3.0
        thrust_std = self.thrust_3sigma / 3.0
        isp_std = self.isp_3sigma / 3.0
        drag_std = self.drag_3sigma / 3.0
        fuel_std = self.fuel_3sigma / 3.0

        for _ in range(self.num_runs):
            d = {
                "thrust_vac_pct": float(self.rng.normal(0.0, thrust_std)),
                "isp_vac_pct": float(self.rng.normal(0.0, isp_std)),
                "drag_cd_pct": float(self.rng.normal(0.0, drag_std)),
                "fuel_mass_pct": float(self.rng.normal(0.0, fuel_std)),
                # Wind gust speed uniformly or Weibull distributed up to wind_max
                "wind_gust_max_mps": float(self.rng.uniform(0.0, self.wind_max)),
                "wind_altitude_min_m": self.wind_alt_min,
                "wind_altitude_max_m": self.wind_alt_max,
                "wind_azimuth_deg": float(self.rng.uniform(0.0, 360.0)),
            }
            dispersions_list.append(d)

        return dispersions_list

    def execute(self, output_csv: str | None = None) -> list[dict[str, Any]]:
        """Execute parallel Monte Carlo simulation across worker processes."""
        dispersions = self.generate_dispersions()
        results: list[dict[str, Any]] = []

        start_time = time.perf_counter()
        print(f"[Monte Carlo] Launching {self.num_runs} runs across {self.max_workers} worker processes...")

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    _single_run_worker,
                    run_id=i + 1,
                    vehicle_cfg_path=self.vehicle_cfg_path,
                    dispersions=dispersions[i],
                    seed=int(self.rng.integers(1, 1000000)),
                )
                for i in range(self.num_runs)
            ]

            completed_count = 0
            for future in as_completed(futures):
                try:
                    res = future.result()
                    results.append(res)
                    completed_count += 1
                    if completed_count % max(1, (self.num_runs // 10)) == 0 or completed_count == self.num_runs:
                        elapsed = time.perf_counter() - start_time
                        print(f"  Progress: {completed_count}/{self.num_runs} completed ({elapsed:.1f}s)")
                except Exception as exc:
                    print(f"[Monte Carlo Error] Worker raised exception: {exc}")

        total_elapsed = time.perf_counter() - start_time
        print(f"[Monte Carlo] All {len(results)} runs completed in {total_elapsed:.2f} seconds.")

        # Sort results by run_id
        results.sort(key=lambda x: x["run_id"])

        # Write to CSV if path specified
        if output_csv:
            os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                if results:
                    fieldnames = list(results[0].keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(results)
            print(f"[Monte Carlo] Results saved to: {output_csv}")

        return results

