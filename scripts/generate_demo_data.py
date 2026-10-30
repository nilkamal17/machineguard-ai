"""Create a clearly labelled, simulated dataset for an offline first run.

This is not real factory telemetry and must not be used to make real maintenance
or safety decisions. Download the UCI AI4I dataset for the main tutorial.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "demo_machine_data.csv"


def generate_demo_data(rows: int = 2500, seed: int = 42) -> pd.DataFrame:
    if rows < 200:
        raise ValueError("Generate at least 200 rows so the demo has enough examples.")

    rng = np.random.default_rng(seed)
    machine_type = rng.choice(["L", "M", "H"], size=rows, p=[0.50, 0.30, 0.20])
    air_temperature = rng.normal(300.0, 2.0, size=rows)
    process_temperature = air_temperature + rng.normal(10.0, 1.0, size=rows)
    rotational_speed = np.clip(rng.normal(1500.0, 220.0, size=rows), 900, 3000)
    torque = np.clip(rng.normal(40.0, 10.0, size=rows), 3, 80)
    tool_wear = np.clip(
        rng.gamma(shape=2.0, scale=58.0, size=rows)
        + np.where(machine_type == "H", 5, np.where(machine_type == "M", 3, 2)),
        0,
        250,
    )

    temperature_gap = process_temperature - air_temperature
    # Illustrative relationship only; it is not a physical failure equation.
    logit = (
        -3.9
        + 0.045 * (torque - 40.0)
        + 0.025 * (tool_wear - 100.0)
        + 0.45 * (temperature_gap - 10.0)
        + 0.0020 * (1500.0 - rotational_speed)
        + 0.25 * (machine_type == "H")
    )
    failure_score = 1.0 / (1.0 + np.exp(-logit))
    machine_failure = rng.binomial(1, failure_score)

    return pd.DataFrame(
        {
            "Type": machine_type,
            "Air temperature [K]": air_temperature.round(2),
            "Process temperature [K]": process_temperature.round(2),
            "Rotational speed [rpm]": rotational_speed.round(0).astype(int),
            "Torque [Nm]": torque.round(2),
            "Tool wear [min]": tool_wear.round(0).astype(int),
            "Machine failure": machine_failure,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline demo machine readings.")
    parser.add_argument("--rows", type=int, default=2500, help="Number of simulated rows (default: 2500).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path.")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = generate_demo_data(rows=args.rows, seed=args.seed)
    data.to_csv(args.output, index=False)
    print(f"Created {len(data):,} simulated rows at: {args.output}")
    print(f"Simulated failure rate: {data['Machine failure'].mean():.2%}")
    print("Reminder: this is demo-only simulated data, not real machine telemetry.")


if __name__ == "__main__":
    main()
