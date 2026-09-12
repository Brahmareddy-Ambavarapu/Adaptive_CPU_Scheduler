"""
Configuration parameters for ML Preprocessing (Step 10).
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DATASET_PATH = DATA_DIR / "cpu_scheduling_dataset.csv"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
LABEL_MAPPING_PATH = MODELS_DIR / "label_mapping.json"

# Preprocessing Constants
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Feature columns (Exactly 10 features, order must be preserved)
FEATURE_COLUMNS = [
    "num_processes",
    "average_burst",
    "burst_std",
    "min_burst",
    "max_burst",
    "arrival_rate",
    "average_priority",
    "priority_std",
    "short_job_ratio",
    "long_job_ratio",
]

# Target column
TARGET_COLUMN = "target_scheduler"

# Explicit Label Mapping (Deterministic)
LABEL_MAPPING = {
    "FCFS": 0,
    "SJF": 1,
    "RR": 2,
    "Priority": 3,
}

# Reverse mapping for convenience
REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}
