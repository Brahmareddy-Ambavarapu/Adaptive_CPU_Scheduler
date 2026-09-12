import os
import sys
import argparse
import subprocess
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from ml.config import FEATURE_COLUMNS, RANDOM_STATE
from ml.model_utils import get_random_forest
from ml.preprocessing import preprocess_dataset

RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARIES_DIR = RESULTS_DIR / "summaries"
EXP_DIR = RESULTS_DIR / "experiments"
DATASET_PATH = PROJECT_ROOT / "data" / "cpu_scheduling_dataset.csv"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.joblib"
MAPPING_PATH = PROJECT_ROOT / "models" / "label_mapping.json"

def run_cpp_scheduler(seed, workloads, out_dir):
    cmd = [
        str(PROJECT_ROOT / "scheduler.exe"),
        "--seed", str(seed),
        "--workloads", str(workloads),
        "--out-dir", str(out_dir)
    ]
    print(f"Running C++ Simulator: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def calc_stats(df, prefix):
    zero_regret = (df[f'{prefix}_regret_percent'] <= 1e-5).mean() * 100
    within_5 = (df[f'{prefix}_regret_percent'] <= 5.0).mean() * 100
    within_10 = (df[f'{prefix}_regret_percent'] <= 10.0).mean() * 100
    
    return {
        "average_regret": df[f'{prefix}_regret_percent'].mean(),
        "median_regret": df[f'{prefix}_regret_percent'].median(),
        "worst_regret": df[f'{prefix}_regret_percent'].max(),
        "zero_regret_percent": zero_regret,
        "within_5_percent": within_5,
        "within_10_percent": within_10
    }

def run_generalization():
    print("=== Generalization Experiment ===")
    eval_seed = 2026
    num_workloads = 20
    out_dir = EXP_DIR / "generalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the C++ simulator with a completely unseen seed
    run_cpp_scheduler(eval_seed, num_workloads, out_dir)
    
    # Parse the results
    csv_path = out_dir / "experiments" / "per_workload.csv"
    df = pd.read_csv(csv_path)
    
    # Calculate classification metrics for ML
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    y_true = df['oracle_scheduler']
    y_pred = df['ml_predicted_scheduler']
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    ml_stats = calc_stats(df, 'ml')
    rb_stats = calc_stats(df, 'rule_based')
    
    res = {
        "experiment": "generalization",
        "eval_seed": eval_seed,
        "num_workloads": num_workloads,
        "ml_accuracy": acc,
        "ml_weighted_precision": prec,
        "ml_weighted_recall": rec,
        "ml_weighted_f1": f1,
        "ml_average_regret": ml_stats['average_regret'],
        "ml_median_regret": ml_stats['median_regret'],
        "ml_worst_regret": ml_stats['worst_regret'],
        "ml_zero_regret_percent": ml_stats['zero_regret_percent'],
        "ml_within_5_percent": ml_stats['within_5_percent'],
        "ml_within_10_percent": ml_stats['within_10_percent'],
        "oracle_average_score": df['oracle_optimal_score'].mean(),
        "ml_average_score": df['ml_score'].mean(),
        "rule_based_average_regret": rb_stats['average_regret'],
        "rule_based_median_regret": rb_stats['median_regret'],
        "rule_based_worst_regret": rb_stats['worst_regret'],
    }
    
    SUMMARIES_DIR.mkdir(exist_ok=True)
    pd.DataFrame([res]).to_csv(SUMMARIES_DIR / "generalization_results.csv", index=False)
    print(f"Results saved to {SUMMARIES_DIR / 'generalization_results.csv'}")
    print(f"ML Generalization Accuracy: {acc:.4f}, Average Regret: {ml_stats['average_regret']:.2f}%")

def run_weights():
    print("=== Objective-Weight Sensitivity ===")
    out_dir = EXP_DIR / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Run baseline to get the raw metrics
    run_cpp_scheduler(42, 20, out_dir)
    csv_path = out_dir / "experiments" / "per_workload.csv"
    df = pd.read_csv(csv_path)
    
    configs = [
        {"name": "Baseline", "w": 0.40, "r": 0.30, "t": 0.20, "cs": 0.10},
        {"name": "Waiting-heavy", "w": 0.60, "r": 0.20, "t": 0.15, "cs": 0.05},
        {"name": "Response-heavy", "w": 0.20, "r": 0.50, "t": 0.20, "cs": 0.10},
        {"name": "Turnaround-heavy", "w": 0.20, "r": 0.20, "t": 0.50, "cs": 0.10},
        {"name": "Context-switch-heavy", "w": 0.25, "r": 0.20, "t": 0.15, "cs": 0.40}
    ]
    
    results = []
    schedulers = ["FCFS", "SJF", "RR", "Priority"]
    
    for c in configs:
        assert abs(c['w'] + c['r'] + c['t'] + c['cs'] - 1.0) < 1e-5, f"Weights must sum to 1.0 in {c['name']}"
        
        oracle_names = []
        ml_regrets = []
        rb_regrets = []
        
        for idx, row in df.iterrows():
            # Find maxes for this workload
            max_w = max([row[f'{s.lower()}_waiting'] for s in schedulers])
            max_r = max([row[f'{s.lower()}_response'] for s in schedulers])
            max_t = max([row[f'{s.lower()}_turnaround'] for s in schedulers])
            max_cs = max([row[f'{s.lower()}_context_switches'] for s in schedulers])
            
            # Recalculate scores for the 4 base schedulers
            scores = {}
            for s in schedulers:
                p = s.lower()
                nw = (row[f'{p}_waiting'] / max_w) if max_w > 0 else 0
                nr = (row[f'{p}_response'] / max_r) if max_r > 0 else 0
                nt = (row[f'{p}_turnaround'] / max_t) if max_t > 0 else 0
                ncs = (row[f'{p}_context_switches'] / max_cs) if max_cs > 0 else 0
                
                score = c['w']*nw + c['r']*nr + c['t']*nt + c['cs']*ncs
                scores[s] = score
                
            # Tie-break Oracle (FCFS > SJF > RR > Priority)
            best_s = schedulers[0]
            best_score = scores[best_s]
            for s in schedulers[1:]:
                if scores[s] < best_score - 1e-9:
                    best_score = scores[s]
                    best_s = s
                    
            oracle_names.append(best_s)
            
            # Recalculate adaptive scores (ML and RB names are given)
            ml_s = row['ml_predicted_scheduler']
            ml_score = scores[ml_s]
            ml_reg = ((ml_score - best_score) / best_score * 100) if best_score > 1e-9 else (0 if ml_score <= 1e-9 else float('inf'))
            ml_regrets.append(ml_reg)
            
            rb_s = row['rule_based_scheduler']
            rb_score = scores[rb_s]
            rb_reg = ((rb_score - best_score) / best_score * 100) if best_score > 1e-9 else (0 if rb_score <= 1e-9 else float('inf'))
            rb_regrets.append(rb_reg)
            
        # Stats for this config
        y_true = oracle_names
        y_pred = df['ml_predicted_scheduler']
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_true, y_pred)
        
        results.append({
            "configuration": c['name'],
            "waiting_weight": c['w'],
            "response_weight": c['r'],
            "turnaround_weight": c['t'],
            "context_switch_weight": c['cs'],
            "ml_accuracy": acc,
            "average_regret": np.mean(ml_regrets),
            "median_regret": np.median(ml_regrets),
            "worst_regret": np.max(ml_regrets),
            "zero_regret_percent": np.mean(np.array(ml_regrets) <= 1e-5) * 100,
            "within_5_percent": np.mean(np.array(ml_regrets) <= 5.0) * 100,
            "within_10_percent": np.mean(np.array(ml_regrets) <= 10.0) * 100
        })
        
    out_csv = SUMMARIES_DIR / "objective_weight_sensitivity.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")


def run_ablation():
    print("=== Feature Ablation Experiment ===")
    out_dir = EXP_DIR / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Read Training Dataset (using the same random state 42 for splitting)
    prep_results = preprocess_dataset(DATASET_PATH, SCALER_PATH, MAPPING_PATH)
    X_train, X_test, y_train, y_test = prep_results['X_train'], prep_results['X_test'], prep_results['y_train'], prep_results['y_test']
    
    features = FEATURE_COLUMNS.copy()
    
    # We will test "none" (full model) and then 10 ablations
    ablations = [None] + features
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import shutil
    
    results = []
    
    for remove_feat in ablations:
        print(f"\nAblating feature: {remove_feat if remove_feat else 'None (Full Model)'}")
        
        # Determine remaining features
        if remove_feat:
            rem_feats = [f for f in features if f != remove_feat]
            # Use raw unscaled dataframe to refit a new scaler
            df_raw = pd.read_csv(DATASET_PATH)
            X_raw = df_raw[features]
            from sklearn.model_selection import train_test_split
            # When running with a small demo size (e.g. 20), stratification will crash if a class has < 2 samples.
            # Therefore, we conditionally use stratification.
            try:
                X_train_raw, X_test_raw, _, _ = train_test_split(X_raw, df_raw["target_scheduler"], test_size=0.2, random_state=42, stratify=df_raw["target_scheduler"])
            except ValueError:
                X_train_raw, X_test_raw, _, _ = train_test_split(X_raw, df_raw["target_scheduler"], test_size=0.2, random_state=42, stratify=None)
            
            X_train_ablated = X_train_raw[rem_feats]
            X_test_ablated = X_test_raw[rem_feats]
            
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_ablated), columns=rem_feats)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test_ablated), columns=rem_feats)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
            rem_feats = features
            scaler = joblib.load(SCALER_PATH)
            
        # Train model (Random Forest was best in Step 11, we use its defaults)
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_m = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        # Evaluate Regret using C++ simulator on a fixed evaluation seed (9999)
        model_path = out_dir / f"model_{remove_feat or 'full'}.joblib"
        scaler_path = out_dir / f"scaler_{remove_feat or 'full'}.joblib"
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        os.environ["ML_MODEL_OVERRIDE"] = str(model_path)
        os.environ["ML_SCALER_OVERRIDE"] = str(scaler_path)
        if remove_feat:
            os.environ["ML_ABLATED_FEATURE"] = remove_feat
        else:
            if "ML_ABLATED_FEATURE" in os.environ:
                del os.environ["ML_ABLATED_FEATURE"]
                
        run_cpp_scheduler(9999, 20, out_dir / (remove_feat or "full"))
        
        # Read the generated CSV
        csv_path = out_dir / (remove_feat or "full") / "experiments" / "per_workload.csv"
        df_sim = pd.read_csv(csv_path)
        
        ml_stats = calc_stats(df_sim, 'ml')
        
        results.append({
            "experiment": "ablation",
            "removed_feature": remove_feat or "none",
            "remaining_features": len(rem_feats),
            "accuracy": acc,
            "weighted_precision": prec,
            "weighted_recall": rec,
            "weighted_f1": f1_w,
            "macro_f1": f1_m,
            "average_regret": ml_stats['average_regret'],
            "median_regret": ml_stats['median_regret'],
            "worst_regret": ml_stats['worst_regret'],
            "zero_regret_percent": ml_stats['zero_regret_percent'],
            "within_5_percent": ml_stats['within_5_percent'],
            "within_10_percent": ml_stats['within_10_percent']
        })
        
    out_csv = SUMMARIES_DIR / "feature_ablation.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")


def run_training_size():
    print("=== Training-Size Sensitivity Experiment ===")
    out_dir = EXP_DIR / "training_size"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    prep_results = preprocess_dataset(DATASET_PATH, SCALER_PATH, MAPPING_PATH)
    X_train, X_test, y_train, y_test = prep_results['X_train'], prep_results['X_test'], prep_results['y_train'], prep_results['y_test']
    
    # Generate fixed evaluation set on seed 10000 for Regret calculation
    eval_seed = 10000
    eval_dir = out_dir / "eval_baseline"
    run_cpp_scheduler(eval_seed, 20, eval_dir)
    csv_path = eval_dir / "experiments" / "per_workload.csv"
    df_sim_base = pd.read_csv(csv_path)
    
    sizes = [100, 250, 500, 750, 1000]
    max_size = len(X_train)
    
    # Include all available sizes
    for s in [2000, 5000, 10000]:
        if s <= max_size:
            sizes.append(s)
            
    if max_size not in sizes:
        sizes.append(max_size)
        
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    results = []
    
    for size in sizes:
        print(f"\nTraining with dataset size: {size}")
        
        # Deterministic subset
        # We can't just take the first N because it might not be stratified. 
        # But for robustness, we just take the first N (since train_test_split shuffles by default in step 10)
        X_train_sub = X_train.iloc[:size]
        y_train_sub = y_train.iloc[:size]
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train_sub, y_train_sub)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_m = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        model_path = out_dir / f"model_size_{size}.joblib"
        joblib.dump(model, model_path)
        
        os.environ["ML_MODEL_OVERRIDE"] = str(model_path)
        if "ML_SCALER_OVERRIDE" in os.environ:
            del os.environ["ML_SCALER_OVERRIDE"]
        if "ML_ABLATED_FEATURE" in os.environ:
            del os.environ["ML_ABLATED_FEATURE"]
            
        run_cpp_scheduler(eval_seed, 20, out_dir / f"size_{size}")
        
        df_sim = pd.read_csv(out_dir / f"size_{size}" / "experiments" / "per_workload.csv")
        ml_stats = calc_stats(df_sim, 'ml')
        
        results.append({
            "training_size": size,
            "accuracy": acc,
            "weighted_precision": prec,
            "weighted_recall": rec,
            "weighted_f1": f1_w,
            "macro_f1": f1_m,
            "average_regret": ml_stats['average_regret'],
            "median_regret": ml_stats['median_regret'],
            "worst_regret": ml_stats['worst_regret'],
            "zero_regret_percent": ml_stats['zero_regret_percent'],
            "within_5_percent": ml_stats['within_5_percent'],
            "within_10_percent": ml_stats['within_10_percent']
        })
        
    out_csv = SUMMARIES_DIR / "training_size_sensitivity.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True, 
                        choices=["generalization", "ablation", "weights", "training-size"])
    args = parser.parse_args()
    
    if args.experiment == "generalization":
        run_generalization()
    elif args.experiment == "weights":
        run_weights()
    elif args.experiment == "ablation":
        run_ablation()
    elif args.experiment == "training-size":
        run_training_size()
    else:
        print(f"Experiment {args.experiment} not implemented yet.")
