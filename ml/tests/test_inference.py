import pytest
import json
import subprocess
from pathlib import Path
from ml.predict_scheduler import predict

def test_predict_function_valid():
    # Provide exactly 10 deterministic float inputs
    features = [10.0, 5.0, 1.0, 1.0, 10.0, 2.0, 2.0, 0.5, 0.5, 0.5]
    result = predict(features)
    
    # Assert no errors
    assert "error" not in result
    assert "predicted_label" in result
    assert "scheduler" in result
    assert "confidence" in result
    
    assert result["predicted_label"] in [0, 1, 2, 3]
    assert result["scheduler"] in ["FCFS", "SJF", "RR", "Priority"]
    # Check if confidence is valid probability or fallback
    assert result["confidence"] == -1.0 or (0.0 <= result["confidence"] <= 1.0)

def test_predict_function_invalid_length():
    features = [10.0, 5.0] # Only 2 features
    result = predict(features)
    assert "error" in result
    assert "exactly 10 features" in result["error"]

def test_predict_function_nan():
    features = [10.0, 5.0, 1.0, 1.0, 10.0, 2.0, float('nan'), 0.5, 0.5, 0.5]
    result = predict(features)
    assert "error" in result
    assert "NaN or Inf" in result["error"]

def test_cli_execution():
    features = ["10.0", "5.0", "1.0", "1.0", "10.0", "2.0", "2.0", "0.5", "0.5", "0.5"]
    script_path = Path(__file__).resolve().parent.parent / "predict_scheduler.py"
    
    result = subprocess.run(
        ["python", str(script_path)] + features,
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    
    # Assert standard output is valid JSON
    output_json = json.loads(result.stdout.strip())
    assert "error" not in output_json
    assert output_json["scheduler"] in ["FCFS", "SJF", "RR", "Priority"]

def test_cli_execution_wrong_args():
    features = ["10.0", "5.0"]
    script_path = Path(__file__).resolve().parent.parent / "predict_scheduler.py"
    
    result = subprocess.run(
        ["python", str(script_path)] + features,
        capture_output=True,
        text=True
    )
    
    # Our script explicitly uses sys.exit(1) for arg length mismatches
    assert result.returncode == 1
    output_json = json.loads(result.stdout.strip())
    assert "error" in output_json
