"""Download the UCI AI4I 2020 Predictive Maintenance dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Connecting to the UCI Machine Learning Repository...")
    dataset = fetch_ucirepo(id=601)

    features = dataset.data.features
    targets = dataset.data.targets
    combined = pd.concat([features.reset_index(drop=True), targets.reset_index(drop=True)], axis=1)

    if "Machine failure" not in combined.columns:
        raise RuntimeError(
            "The downloaded dataset did not contain 'Machine failure'. "
            "Please check the UCI dataset page and column names."
        )

    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Downloaded {len(combined):,} rows.")
    print(f"Saved dataset to: {OUTPUT_PATH}")
    print("Source: UCI AI4I 2020 Predictive Maintenance Dataset (CC BY 4.0).")
    print("Note: UCI describes this dataset as synthetic, designed to reflect industrial data.")


if __name__ == "__main__":
    main()
