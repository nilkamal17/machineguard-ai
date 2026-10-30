"""Dataset loading, column normalization, and input validation."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET_COLUMN, NUMERIC_FEATURES


def _normalise_name(value: object) -> str:
    name = str(value).strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", name)


_COLUMN_ALIASES = {
    "type": "Type",
    "air temperature [k]": "Air temperature [K]",
    "air temperature": "Air temperature [K]",
    "air temperature k": "Air temperature [K]",
    "air temperature (k)": "Air temperature [K]",
    "process temperature [k]": "Process temperature [K]",
    "process temperature": "Process temperature [K]",
    "process temperature k": "Process temperature [K]",
    "process temperature (k)": "Process temperature [K]",
    "rotational speed [rpm]": "Rotational speed [rpm]",
    "rotational speed": "Rotational speed [rpm]",
    "rpm": "Rotational speed [rpm]",
    "torque [nm]": "Torque [Nm]",
    "torque": "Torque [Nm]",
    "tool wear [min]": "Tool wear [min]",
    "tool wear": "Tool wear [min]",
    "machine failure": TARGET_COLUMN,
    "failure": TARGET_COLUMN,
}


def canonicalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map common header variations to the official AI4I column names."""
    rename_map = {}
    for column in frame.columns:
        normalised = _normalise_name(column)
        if normalised in _COLUMN_ALIASES:
            rename_map[column] = _COLUMN_ALIASES[normalised]
    result = frame.rename(columns=rename_map).copy()

    duplicated = result.columns[result.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(
            "More than one uploaded column maps to the same field: "
            + ", ".join(sorted(set(duplicated)))
        )
    return result


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the six model input columns in the correct order.

    Additional CSV columns are ignored. Missing feature values are retained as
    NaN because the sklearn pipeline imputes them during prediction.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("The input data is empty. Upload a CSV with machine readings.")

    result = canonicalise_columns(frame)
    missing = [column for column in FEATURE_COLUMNS if column not in result.columns]
    if missing:
        raise ValueError(
            "Missing required sensor columns: " + ", ".join(missing)
            + ". See the README for the exact CSV format."
        )

    features = result[FEATURE_COLUMNS].copy()
    features["Type"] = features["Type"].astype("string").str.strip().str.upper()
    features.loc[features["Type"].isin(["", "NAN", "NONE"]), "Type"] = pd.NA

    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    return features


def load_training_data(path: str | Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load a labelled CSV and return clean model features and binary target."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Run scripts/download_data.py first, "
            "or use data/demo_machine_data.csv."
        )

    raw = pd.read_csv(csv_path)
    raw = canonicalise_columns(raw)
    if TARGET_COLUMN not in raw.columns:
        raise ValueError(
            f"Training data must contain the target column '{TARGET_COLUMN}'."
        )

    features = prepare_features(raw)
    target = pd.to_numeric(raw[TARGET_COLUMN], errors="coerce")
    valid_target = target.isin([0, 1])
    features = features.loc[valid_target].reset_index(drop=True)
    target = target.loc[valid_target].astype(int).reset_index(drop=True)

    if features.empty:
        raise ValueError("No valid labelled rows found. Target values must be 0 or 1.")
    if target.nunique() < 2:
        raise ValueError("Training data must contain both failure (1) and no-failure (0) rows.")

    return features, target
