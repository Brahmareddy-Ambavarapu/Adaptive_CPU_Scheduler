import json
import math
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from .config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    LABEL_MAPPING,
    RANDOM_STATE,
    TEST_SIZE,
    MODELS_DIR,
)

def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validates the dataset according to the strict Step 10 rules.
    Raises ValueError if validation fails.
    """
    # A. Required columns exist
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
            
    # B. No duplicate column names
    if len(df.columns) != len(set(df.columns)):
        raise ValueError("Dataset contains duplicate column names.")
        
    # C. Feature columns are numeric
    for col in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Feature column '{col}' is not numeric.")
            
    # D, E. No missing or NaN values
    if df.isna().any().any():
        raise ValueError("Dataset contains missing or NaN values.")
        
    # F. No positive or negative infinity
    for col in FEATURE_COLUMNS:
        if np.isinf(df[col]).any():
            raise ValueError(f"Feature column '{col}' contains infinite values.")
            
    # G. target_scheduler contains ONLY valid classes
    invalid_targets = set(df[TARGET_COLUMN].unique()) - set(LABEL_MAPPING.keys())
    if invalid_targets:
        raise ValueError(f"Invalid target classes found: {invalid_targets}")
        
    # H. num_processes is positive
    if not (df["num_processes"] > 0).all():
        raise ValueError("num_processes must be positive.")
        
    # I. burst-related values are valid
    if not (df["average_burst"] > 0).all():
        raise ValueError("average_burst must be positive.")
    if not (df["min_burst"] > 0).all():
        raise ValueError("min_burst must be positive.")
    if not (df["max_burst"] > 0).all():
        raise ValueError("max_burst must be positive.")
    if not (df["min_burst"] <= df["average_burst"]).all() or not (df["average_burst"] <= df["max_burst"]).all():
         raise ValueError("min_burst <= average_burst <= max_burst relation violated.")
         
    # J. burst_std >= 0
    if not (df["burst_std"] >= 0).all():
        raise ValueError("burst_std must be non-negative.")
        
    # K. arrival_rate is finite and non-negative
    if not (df["arrival_rate"] >= 0).all():
        raise ValueError("arrival_rate must be non-negative.")
        
    # L. average_priority is valid (1 is highest, assuming >= 1)
    if not (df["average_priority"] >= 1).all():
        raise ValueError("average_priority must be >= 1.")
        
    # M. priority_std >= 0
    if not (df["priority_std"] >= 0).all():
        raise ValueError("priority_std must be non-negative.")
        
    # N. short_job_ratio is between 0 and 1
    if not ((df["short_job_ratio"] >= 0) & (df["short_job_ratio"] <= 1)).all():
        raise ValueError("short_job_ratio must be between 0 and 1.")
        
    # O. long_job_ratio is between 0 and 1
    if not ((df["long_job_ratio"] >= 0) & (df["long_job_ratio"] <= 1)).all():
        raise ValueError("long_job_ratio must be between 0 and 1.")
        
    # P. short_job_ratio + long_job_ratio should be approximately 1.0
    ratio_sum = df["short_job_ratio"] + df["long_job_ratio"]
    if not np.allclose(ratio_sum, 1.0, atol=1e-5):
        raise ValueError("short_job_ratio + long_job_ratio must equal approximately 1.0.")

def load_dataset(path: str | Path) -> pd.DataFrame:
    """Loads and validates the dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Please run Step 9 dataset generator first.")
        
    df = pd.read_csv(path)
    validate_dataset(df)
    return df

def split_dataset(df: pd.DataFrame):
    """
    Splits the dataset into train and test sets at the workload level.
    Returns X_train, X_test, y_train, y_test preserving feature order.
    """
    # Isolate exact 10 features, preventing leakage
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    
    # Split 80/20 with stratification and fixed random state
    # Use stratification by target_scheduler where valid (i.e. all classes >= 2)
    stratify_arg = y if y.value_counts().min() >= 2 else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=stratify_arg
    )
    
    return X_train, X_test, y_train, y_test

def encode_target(y: pd.Series) -> pd.Series:
    """Encodes string target labels to integers deterministically."""
    return y.map(LABEL_MAPPING)

def preprocess_dataset(dataset_path: str | Path, scaler_path: str | Path, mapping_path: str | Path):
    """
    Full pipeline: load, validate, split, encode, fit scaler on train, transform, save artifacts.
    """
    df = load_dataset(dataset_path)
    
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = split_dataset(df)
    
    y_train = encode_target(y_train_raw)
    y_test = encode_target(y_test_raw)
    
    # Fit StandardScaler ONLY on training data to prevent leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # Convert back to DataFrame to keep feature names
    X_train = pd.DataFrame(X_train_scaled, columns=FEATURE_COLUMNS, index=X_train_raw.index)
    X_test = pd.DataFrame(X_test_scaled, columns=FEATURE_COLUMNS, index=X_test_raw.index)
    
    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save artifacts
    joblib.dump(scaler, scaler_path)
    with open(mapping_path, 'w') as f:
        json.dump(LABEL_MAPPING, f, indent=4)
        
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_raw": y_train_raw,
        "y_test_raw": y_test_raw,
        "df_raw": df
    }
