from pathlib import Path

import pandas as pd
import pytest

from src.data_utils import canonicalise_columns, prepare_features, load_training_data
from src.config import FEATURE_COLUMNS
from scripts.generate_demo_data import generate_demo_data


def test_demo_generator_has_both_classes_and_expected_columns():
    data = generate_demo_data(rows=1200, seed=7)
    assert set(FEATURE_COLUMNS).issubset(data.columns)
    assert "Machine failure" in data.columns
    assert set(data["Machine failure"].unique()).issubset({0, 1})
    assert data["Machine failure"].nunique() == 2


def test_header_aliases_are_normalised():
    frame = pd.DataFrame(
        {
            "type": ["M"],
            "air_temperature": [300.0],
            "process_temperature": [310.0],
            "rpm": [1500],
            "torque": [40.0],
            "tool_wear": [100],
        }
    )
    prepared = prepare_features(frame)
    assert list(prepared.columns) == FEATURE_COLUMNS
    assert prepared.iloc[0]["Rotational speed [rpm]"] == 1500


def test_missing_required_sensor_is_reported():
    with pytest.raises(ValueError, match="Missing required sensor columns"):
        prepare_features(pd.DataFrame({"Type": ["M"]}))


def test_load_training_data_filters_invalid_target():
    data = generate_demo_data(rows=1200, seed=12)
    data.loc[0, "Machine failure"] = 9
    temp_path = Path("test_training_data.csv")
    try:
        data.to_csv(temp_path, index=False)
        X, y = load_training_data(temp_path)
        assert len(X) == 1199
        assert len(y) == 1199
        assert set(y.unique()) == {0, 1}
    finally:
        temp_path.unlink(missing_ok=True)
