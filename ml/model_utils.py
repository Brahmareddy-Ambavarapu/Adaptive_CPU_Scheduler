import json
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from .config import RANDOM_STATE, MODELS_DIR

def get_logistic_regression():
    return LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    )

def get_decision_tree():
    return DecisionTreeClassifier(
        random_state=RANDOM_STATE
    )

def get_random_forest():
    return RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

def save_model(model, filename: str):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    return path

def load_model(filename: str):
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

def save_metadata(metadata: dict, filename: str = "model_metadata.json"):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=4)
    return path
