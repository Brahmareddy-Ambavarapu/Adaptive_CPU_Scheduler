import pytest
import pandas as pd
import numpy as np
import tempfile
import json
import joblib
from pathlib import Path

from ml.model_utils import (
    get_logistic_regression, get_decision_tree, get_random_forest,
    save_model, load_model, save_metadata
)
from ml.evaluate_models import evaluate_model
from ml.config import FEATURE_COLUMNS, LABEL_MAPPING, RANDOM_STATE

@pytest.fixture
def synthetic_data():
    """Generates small synthetic train/test sets."""
    np.random.seed(RANDOM_STATE)
    X_train = pd.DataFrame(np.random.randn(20, 10), columns=FEATURE_COLUMNS)
    y_train = pd.Series(np.random.choice([0, 1, 2, 3], size=20))
    X_test = pd.DataFrame(np.random.randn(5, 10), columns=FEATURE_COLUMNS)
    y_test = pd.Series(np.random.choice([0, 1, 2, 3], size=5))
    return X_train, y_train, X_test, y_test

def test_model_constructors():
    lr = get_logistic_regression()
    dt = get_decision_tree()
    rf = get_random_forest()
    
    assert lr is not None
    assert dt is not None
    assert rf is not None
    assert lr.random_state == RANDOM_STATE
    assert dt.random_state == RANDOM_STATE
    assert rf.random_state == RANDOM_STATE

def test_model_training_and_prediction(synthetic_data):
    X_train, y_train, X_test, y_test = synthetic_data
    
    models = [get_logistic_regression(), get_decision_tree(), get_random_forest()]
    for model in models:
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        
        # Predictions are valid labels
        for p in preds:
            assert p in [0, 1, 2, 3]
            
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            # Expecting shape (n_samples, n_classes_seen_in_train)
            assert probs.shape[0] == len(X_test)
            assert probs.shape[1] <= 4 # Could be less if some classes missing in synthetic train
            
            # Probabilities sum to 1
            assert np.allclose(probs.sum(axis=1), 1.0)
            # Between 0 and 1
            assert np.all((probs >= 0) & (probs <= 1))

def test_evaluation_metrics():
    y_true = np.array([0, 1, 2, 3, 0])
    y_pred = np.array([0, 1, 2, 2, 0]) # 1 mistake on class 3
    
    res = evaluate_model("TestModel", y_true, y_pred)
    assert res["model"] == "TestModel"
    assert "accuracy" in res
    assert "weighted_precision" in res
    assert "weighted_recall" in res
    assert "weighted_f1" in res
    assert "macro_f1" in res
    assert res["accuracy"] == 0.8 # 4 out of 5

def test_serialization():
    model = get_logistic_regression()
    with tempfile.TemporaryDirectory() as tmpdir:
        from ml.model_utils import MODELS_DIR
        # Temporarily mock MODELS_DIR
        import ml.model_utils
        orig_dir = ml.model_utils.MODELS_DIR
        ml.model_utils.MODELS_DIR = Path(tmpdir)
        
        path = save_model(model, "test_lr.joblib")
        assert path.exists()
        
        loaded = load_model("test_lr.joblib")
        assert loaded.random_state == model.random_state
        
        meta_path = save_metadata({"test": "data"}, "meta.json")
        assert meta_path.exists()
        with open(meta_path) as f:
             assert json.load(f)["test"] == "data"
             
        # Restore
        ml.model_utils.MODELS_DIR = orig_dir

def test_feature_importance_rf(synthetic_data):
    X_train, y_train, X_test, y_test = synthetic_data
    rf = get_random_forest()
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    assert len(importances) == 10
    assert np.all(importances >= 0)
