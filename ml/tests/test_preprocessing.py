import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import json
import joblib

from ml.config import FEATURE_COLUMNS, TARGET_COLUMN, LABEL_MAPPING, RANDOM_STATE
from ml.preprocessing import (
    validate_dataset,
    load_dataset,
    split_dataset,
    encode_target,
    preprocess_dataset
)

@pytest.fixture
def valid_dataframe():
    """Generates a minimal valid dataframe for testing."""
    data = {
        "workload_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "num_processes": [10] * 10,
        "average_burst": [5.0] * 10,
        "burst_std": [2.0] * 10,
        "min_burst": [1.0] * 10,
        "max_burst": [10.0] * 10,
        "arrival_rate": [0.1] * 10,
        "average_priority": [2.0] * 10,
        "priority_std": [1.0] * 10,
        "short_job_ratio": [0.5] * 10,
        "long_job_ratio": [0.5] * 10,
        "target_scheduler": ["FCFS", "SJF", "RR", "Priority", "FCFS", "SJF", "RR", "Priority", "FCFS", "SJF"],
        "unrelated_metric": [999] * 10
    }
    return pd.DataFrame(data)


def test_required_columns(valid_dataframe):
    df = valid_dataframe.copy()
    validate_dataset(df)  # Should not raise

    df = df.drop(columns=["num_processes"])
    with pytest.raises(ValueError, match="Missing required column"):
        validate_dataset(df)


def test_duplicate_columns(valid_dataframe):
    df = pd.concat([valid_dataframe, valid_dataframe["num_processes"]], axis=1)
    # pandas concat adds duplicate column name 'num_processes'
    with pytest.raises(ValueError, match="duplicate column names"):
         validate_dataset(df)


def test_numeric_features(valid_dataframe):
    df = valid_dataframe.copy()
    df["average_burst"] = "not a number"
    with pytest.raises(ValueError, match="is not numeric"):
        validate_dataset(df)


def test_missing_values(valid_dataframe):
    df = valid_dataframe.copy()
    df.loc[0, "average_burst"] = None
    with pytest.raises(ValueError, match="missing or NaN values"):
        validate_dataset(df)


def test_nan_values(valid_dataframe):
    df = valid_dataframe.copy()
    df.loc[0, "average_burst"] = np.nan
    with pytest.raises(ValueError, match="missing or NaN values"):
        validate_dataset(df)


def test_infinite_values(valid_dataframe):
    df = valid_dataframe.copy()
    df.loc[0, "average_burst"] = np.inf
    with pytest.raises(ValueError, match="contains infinite values"):
        validate_dataset(df)


def test_invalid_target_label(valid_dataframe):
    df = valid_dataframe.copy()
    df.loc[0, "target_scheduler"] = "UNKNOWN_SCHEDULER"
    with pytest.raises(ValueError, match="Invalid target classes"):
        validate_dataset(df)


def test_num_processes_positive(valid_dataframe):
    df = valid_dataframe.copy()
    df.loc[0, "num_processes"] = 0
    with pytest.raises(ValueError, match="num_processes must be positive"):
        validate_dataset(df)


def test_burst_relationships(valid_dataframe):
    df = valid_dataframe.copy()
    # Test average_burst > 0
    df.loc[0, "average_burst"] = -1
    with pytest.raises(ValueError, match="average_burst must be positive"):
        validate_dataset(df)

    df = valid_dataframe.copy()
    # min_burst <= average_burst <= max_burst violation
    df.loc[0, "min_burst"] = 6.0
    df.loc[0, "average_burst"] = 5.0
    with pytest.raises(ValueError, match="min_burst <= average_burst <= max_burst relation violated"):
        validate_dataset(df)


def test_invalid_ratios(valid_dataframe):
    df = valid_dataframe.copy()
    df.loc[0, "short_job_ratio"] = 1.5
    with pytest.raises(ValueError, match="short_job_ratio must be between 0 and 1"):
        validate_dataset(df)

    df = valid_dataframe.copy()
    df.loc[0, "short_job_ratio"] = 0.4
    df.loc[0, "long_job_ratio"] = 0.4
    # sum is 0.8
    with pytest.raises(ValueError, match=r"short_job_ratio \+ long_job_ratio must equal approximately 1.0"):
        validate_dataset(df)


def test_split_dataset(valid_dataframe):
    # Need enough data for stratify to work (at least 2 per class in test size, 
    # but valid_dataframe has 10 rows. Stratify might fail if classes < 2 in test.
    # Let's create a slightly larger dataframe just for split test.
    
    data = {
        "workload_id": list(range(100)),
        "target_scheduler": ["FCFS", "SJF", "RR", "Priority"] * 25
    }
    for col in FEATURE_COLUMNS:
        data[col] = [1.0] * 100
        
    df = pd.DataFrame(data)
    
    X_train, X_test, y_train, y_test = split_dataset(df)
    
    assert len(X_train) == 80
    assert len(X_test) == 20
    
    # 3. Exactly 10 ML features are selected
    assert X_train.shape[1] == 10
    
    # 4. Feature order is correct
    assert list(X_train.columns) == FEATURE_COLUMNS
    
    # 5, 6, 7. workload_id, target_scheduler, and performance metrics are NOT in X
    assert "workload_id" not in X_train.columns
    assert "target_scheduler" not in X_train.columns


def test_deterministic_split(valid_dataframe):
    data = {
        "workload_id": list(range(100)),
        "target_scheduler": ["FCFS", "SJF", "RR", "Priority"] * 25
    }
    for col in FEATURE_COLUMNS:
        data[col] = [1.0] * 100
    df = pd.DataFrame(data)

    X_train1, _, _, _ = split_dataset(df)
    X_train2, _, _, _ = split_dataset(df)
    
    pd.testing.assert_frame_equal(X_train1, X_train2)


def test_label_mapping():
    y = pd.Series(["FCFS", "SJF", "RR", "Priority"])
    y_encoded = encode_target(y)
    
    assert y_encoded.iloc[0] == 0
    assert y_encoded.iloc[1] == 1
    assert y_encoded.iloc[2] == 2
    assert y_encoded.iloc[3] == 3


def test_scaler_fitted_only_on_training_data():
    """
    Construct a dataset where fitting on the complete dataset 
    would produce a different mean/scale than fitting only on training data.
    """
    # Create 100 workloads. 
    # Train set (80) has average_burst = 10
    # Test set (20) has average_burst = 1000
    # If scaler is fitted on ALL data, mean will be around ~200.
    # If scaler is fitted on TRAIN only, mean will be 10.
    
    # To control train/test split exactly, we will mimic the preprocessing pipeline behavior
    # but inject specific data that we know how `train_test_split` with `random_state=42` will split.
    
    data = {
        "target_scheduler": ["FCFS", "SJF", "RR", "Priority"] * 25
    }
    for col in FEATURE_COLUMNS:
        data[col] = [10.0] * 100
    
    data["short_job_ratio"] = [0.5] * 100
    data["long_job_ratio"] = [0.5] * 100
        
    df = pd.DataFrame(data)
    
    X_train_raw, X_test_raw, y_train, y_test = split_dataset(df)
    
    # Now artificially alter the raw df to have massive values for the test indices
    test_indices = X_test_raw.index
    df.loc[test_indices, "average_burst"] = 1000.0
    df.loc[test_indices, "max_burst"] = 1000.0
    
    # Repass this through the pipeline (simulated)
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "data.csv"
        df.to_csv(dataset_path, index=False)
        
        scaler_path = Path(tmpdir) / "scaler.joblib"
        mapping_path = Path(tmpdir) / "mapping.json"
        
        res = preprocess_dataset(dataset_path, scaler_path, mapping_path)
        
        X_train_scaled = res["X_train"]
        X_test_scaled = res["X_test"]
        
        # If fitted only on train (mean=10, std=0 since all 10), it might divide by zero or result in 0
        # Wait, if std=0, StandardScaler sets it to 1.
        # Let's make it have variance in train.
        
        df.loc[X_train_raw.index[:40], "average_burst"] = 5.0
        df.loc[X_train_raw.index[:40], "min_burst"] = 5.0
        df.loc[X_train_raw.index[40:], "average_burst"] = 15.0
        df.loc[X_train_raw.index[40:], "max_burst"] = 15.0
        # Train mean = 10.0
        
        df.to_csv(dataset_path, index=False)
        res = preprocess_dataset(dataset_path, scaler_path, mapping_path)
        X_train_scaled = res["X_train"]
        
        # Mean should be ~0 and std should be ~1 for train
        assert np.isclose(X_train_scaled["average_burst"].mean(), 0, atol=1e-7)
        assert np.isclose(X_train_scaled["average_burst"].std(), 1.0, atol=1e-1)


def test_artifacts_saved(valid_dataframe):
    # Expand dataframe for stratify
    data = {
        "workload_id": list(range(100)),
        "target_scheduler": ["FCFS", "SJF", "RR", "Priority"] * 25
    }
    for col in FEATURE_COLUMNS:
        data[col] = [1.0] * 100
    data["short_job_ratio"] = [0.5] * 100
    data["long_job_ratio"] = [0.5] * 100
    df = pd.DataFrame(data)
    # Avoid zero variance for scaling to ensure no nan/inf in scaled output
    df["average_burst"] = np.random.uniform(2.0, 5.0, 100)
    df["min_burst"] = 1.0
    df["max_burst"] = 10.0
    
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "data.csv"
        df.to_csv(dataset_path, index=False)
        
        scaler_path = Path(tmpdir) / "scaler.joblib"
        mapping_path = Path(tmpdir) / "mapping.json"
        
        res = preprocess_dataset(dataset_path, scaler_path, mapping_path)
        
        assert scaler_path.exists()
        assert mapping_path.exists()
        
        # Check mapping content
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        assert mapping == LABEL_MAPPING
        
        # Check joblib load
        scaler = joblib.load(scaler_path)
        assert hasattr(scaler, "transform")
        
        # No NaN or infinity
        assert not res["X_train"].isna().any().any()
        assert not np.isinf(res["X_train"]).any().any()
