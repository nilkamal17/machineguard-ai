"""Shared project configuration."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

DEFAULT_DATA_PATH = DATA_DIR / "ai4i2020.csv"
DEMO_DATA_PATH = DATA_DIR / "demo_machine_data.csv"
MODEL_PATH = MODEL_DIR / "machine_failure_model.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"

FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET_COLUMN = "Machine failure"
CATEGORICAL_FEATURES = ["Type"]
NUMERIC_FEATURES = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_FEATURES]
DEFAULT_ALERT_THRESHOLD = 0.25
RANDOM_STATE = 42
